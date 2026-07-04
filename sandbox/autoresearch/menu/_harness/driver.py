"""Per-loop stateless ratchet driver (DESIGN_DOC §6).

One loop = one nested git repo. Every iteration: assemble+validate context → agent edits train.py
→ scope fence → CPU smoke → commit the attempt → score (GPU) → keep (advance `best` tag) or reset
(restore champion working file) → journal + distiller. EVERY attempt commits, so `git log/show/diff`
expose the exact diff of every iteration (kept and failed); `best` tag tracks the champion.

`scorer(loop, train_py) -> (era_score, aspects, status, meta)` and
`agent(loop, bundle)   -> (change_summary, reflect_fields)` are injected, so the same ratchet runs
with the real GPU scorer + claude/cursor agent (agent_step.py) or fast mocks (G4 test).

NEVER STOP lives here: `run_loop` keeps going until a stop flag appears — there is no prompt-level
"don't ask the human"; the driver simply never self-halts.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import context
import journal
import distiller
import scope as scope_mod

BEST_REF = context.BEST_REF
FAIL_SCORE = -1e9
RATE_LIMIT_RECHECK_S = int(os.environ.get("MENU_RATELIMIT_RECHECK_S", "1800"))  # poll for limit reset
_ASPECT_MAP = {  # ERA_ASPECTS footer key -> results.tsv column
    "S_A": "S_A", "Q_imp": "Q_imp", "Q_den": "Q_den", "ece": "ece", "dcr": "dcr",
    "c_index": "c_index", "peak_auroc": "peak_auroc",
    "imp_pval_spearman": "imp_pval_sp", "imp_pval_pearson": "imp_pval_pe",
    "imp_count_spearman": "imp_count_sp", "imp_count_pearson": "imp_count_pe",
}


def _git(loop, *a, check=True):
    return subprocess.run(["git", "-C", str(loop), *a], check=check, capture_output=True, text=True)


def _commit(loop, msg):
    _git(loop, "add", "-A")
    _git(loop, "-c", "user.email=ar@menu", "-c", "user.name=menu-ar", "commit", "-q",
         "--allow-empty", "-m", msg)
    return _git(loop, "rev-parse", "--short", "HEAD").stdout.strip()


def parse_footer(text: str) -> tuple[float, dict, str]:
    """Parse `ERA_SCORE:` + `ERA_ASPECTS:` + the harness train line from a run's stdout."""
    sc = re.findall(r"ERA_SCORE:\s*([-+0-9.eE]+)", text)
    if not sc:
        return FAIL_SCORE, {}, "crash"
    score = float(sc[-1])
    aspects = {}
    m = re.search(r"ERA_ASPECTS:\s*(\{.*\})", text)
    if m:
        try:
            raw = json.loads(m.group(1))
            aspects = {col: raw.get(k) for k, col in _ASPECT_MAP.items()}
        except json.JSONDecodeError:
            pass
    tm = re.search(r"trained (\d+) steps \(([\d.]+) epochs\) in ([\d.]+)s", text)
    if tm:
        aspects.update(steps=tm.group(1), epochs=tm.group(2), sec=tm.group(3))
    status = "ok" if score > FAIL_SCORE else "crash"
    return score, aspects, status


# --------------------------------------------------------------------------- loop lifecycle
def init_loop(loop: Path, row0_train: Path, program_md: str, base_score: float, base_aspects: dict) -> None:
    """Iteration 0: candi_v2 as-is is the base commit + champion. base_score from the G1 run."""
    loop = Path(loop)
    loop.mkdir(parents=True, exist_ok=True)
    (loop / "train.py").write_text(Path(row0_train).read_text())
    # vendored, EDITABLE model package (fork of candi_v2): each loop owns its own copy so an
    # agent can rewrite architecture (latent un-detached, attention in encoder.py editable).
    shutil.copytree(Path(row0_train).parent / "candi_model", loop / "candi_model",
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (loop / "program.md").write_text(program_md)
    (loop / "run.log").write_text("")
    (loop / "runs").mkdir(exist_ok=True)
    # git tracks train.py + candi_model/ + program.md + .gitignore → the diff history is the code
    # lineage, uncluttered by the runtime memory files (the SEPARATE journal the context reads).
    (loop / ".gitignore").write_text(
        "results.tsv\nreflections.md\nbacklog.md\nrun.log\nruns/\nrun_driver.out\n"
        "*.flag\n*.out\n.smoke_batch.pt\n__pycache__/\n*.pyc\n")
    _git(loop, "init", "-q")
    _commit(loop, "iter 0 base (candi_v2 as-is)")
    _git(loop, "tag", "-f", BEST_REF)
    journal.append_result(loop, dict(iter=0, ts=time.strftime("%Y%m%dT%H%M%S"), commit="base",
                                     parent="-", status="base", era_score=f"{base_score:.4f}",
                                     d_best="-", change_summary="candi_v2 as-is (row-0)", **base_aspects))
    distiller.update_backlog(loop, 0)


def _real_smoke(loop: Path, timeout_s: int = 300) -> bool:
    """CPU preflight. Always logs stdout+stderr+timing to <loop>/_last_smoke.log so a failure is
    never a black box. timeout=300s (login-node load + the lockstep label iter's first h5 read can
    push the tiny smoke past 180s without the candidate being broken)."""
    t0 = time.time()
    smoke = str(Path(__file__).parent / "smoke.py")
    try:
        sm = subprocess.run(["python", smoke, str(loop / "train.py")],
                            capture_output=True, text=True, timeout=timeout_s)
        ok = "SMOKE_OK" in sm.stdout
        (loop / "runs" / "last_smoke.log").write_text(
            f"[{time.time()-t0:.0f}s] ok={ok}\n--- stdout ---\n{sm.stdout}\n--- stderr ---\n{sm.stderr}")
        return ok
    except subprocess.TimeoutExpired as e:
        subprocess.run(["pkill", "-9", "-u", "mforooz", "-f", "smoke.py"], check=False)
        (loop / "runs" / "last_smoke.log").write_text(
            f"[TIMEOUT >{timeout_s}s]\n--- partial stdout ---\n{e.stdout or ''}\n--- stderr ---\n{e.stderr or ''}")
        return False


def _recover(loop: Path) -> None:
    """Idempotent crash reconcile. The watchdog may restart a driver that died mid-iteration — most
    likely during the ~20-min GPU scoring wait, AFTER the attempt was committed but BEFORE keep/reset
    + journaling. That leaves working≠champion (and possibly a half-written journal), which would make
    context.validate fail forever → a no-progress git-spin. This makes the loop self-consistent again
    so the next iteration just resumes from the champion. A NO-OP on an already-clean loop.
      - working train.py/candi_model  -> champion  (discard the un-scored orphan attempt)
      - reflections padded to match non-base result rows; backlog re-stamped to the last recorded iter
    """
    rc = subprocess.run(["git", "-C", str(loop), "rev-parse", "--verify", BEST_REF],
                        capture_output=True, text=True).returncode
    if rc != 0:
        return                                   # no champion yet (pre-base)
    _git(loop, "checkout", BEST_REF, "--", "train.py", "candi_model")   # working <- champion
    rows = journal.read_results(loop)
    if not rows:
        return
    last_it = max(int(r["iter"]) for r in rows)
    nonbase = sum(1 for r in rows if r.get("status") != "base")
    while len(journal.read_reflections(loop)) < nonbase:               # interrupted mid-journal
        journal.append_reflection(loop, last_it, "crash", "—", "—",
                                  {"result": "[recovered] driver interrupted mid-journal",
                                   "interpretation": "reconciled"})
    distiller.update_backlog(loop, last_it)


def run_iteration(loop: Path, scorer, agent, smoke_fn=_real_smoke) -> dict:
    loop = Path(loop)
    _recover(loop)                       # crash reconcile (idempotent; no-op on a clean loop)
    rows = journal.read_results(loop)
    it = max(int(r["iter"]) for r in rows) + 1
    best = journal.best_row(loop)
    best_score = float(best["era_score"])
    parent = _git(loop, "rev-parse", "--short", "HEAD").stdout.strip()

    # 1. assemble + VALIDATE context (no token spent on a bad bundle)
    bundle = context.assemble(loop)
    ok, problems = context.validate(loop, bundle)
    if not ok:
        return {"iter": it, "status": "ctx_error", "problems": problems}

    # 2. agent edits train.py (returns its pre-score reasoning)
    change_summary, fields = agent(loop, bundle)

    # 2b. NO edit produced (both agents failed) → do NOT score (re-running the unchanged champion on a
    # GPU is pure waste) and do NOT commit/advance/journal. Working tree is clean → nothing to revert.
    # run_loop decides whether to wait (rate-limited) or back off (plain skip).
    outcome = fields.get("outcome")
    if outcome in ("rate_limited", "no_edit"):
        return {"iter": it, "status": outcome}

    # 3. scope fence
    sok, offenders = scope_mod.check_scope(loop)
    if not sok:
        _git(loop, "checkout", BEST_REF, "--", "train.py", "candi_model")
        return _record(loop, it, parent, "gate", best_score, best_score, change_summary,
                       {**fields, "result": f"scope violation {offenders}", "interpretation": "reverted"}, {})

    # 4. CPU smoke preflight
    if not smoke_fn(loop):
        (loop / "runs").mkdir(exist_ok=True)
        (loop / "runs" / f"iter{it:04d}.smokefail.py").write_text((loop / "train.py").read_text())
        _git(loop, "checkout", BEST_REF, "--", "train.py", "candi_model")
        return _record(loop, it, parent, "crash", best_score, best_score, change_summary,
                       {**fields, "result": "smoke failed (preflight)", "interpretation": "reverted"},
                       {"smoke": "fail"})

    # 5. commit the attempt (preserves the exact diff in git history)
    commit = _commit(loop, f"iter {it}: {change_summary[:60]}")

    # 6. score (GPU)
    score, aspects, st = scorer(loop, loop / "train.py")
    status = "crash" if st != "ok" else ("keep" if score > best_score else "reset")

    # 7. keep → advance champion; reset/crash → restore champion working file
    if status == "keep":
        _git(loop, "tag", "-f", BEST_REF)
    else:
        _git(loop, "checkout", BEST_REF, "--", "train.py", "candi_model")

    new_best = max(best_score, score)
    fields = {**fields, "result": f"era_score {score:.4f} (best {new_best:.4f})",
              "interpretation": {"keep": "kept (new champion)", "reset": "reset (did not beat best)",
                                 "crash": "crash/gate → −1e9, reset"}[status]}
    return _record(loop, it, parent, status, score, new_best, change_summary, fields,
                   {**aspects, "commit": commit, "smoke": "pass"})


def _record(loop, it, parent, status, score, new_best, change_summary, fields, aspects) -> dict:
    change_summary = " ".join(str(change_summary).split())   # TSV-safe: no stray tab/newline in the row
    d = score - new_best
    journal.append_result(loop, dict(
        iter=it, ts=time.strftime("%Y%m%dT%H%M%S"), commit=aspects.get("commit", "-"), parent=parent,
        status=status, era_score=f"{score:.4f}", d_best=f"{d:+.4f}", change_summary=change_summary,
        **{k: aspects.get(k, "") for k in
           ("S_A", "Q_imp", "Q_den", "ece", "dcr", "c_index", "peak_auroc",
            "imp_pval_sp", "imp_pval_pe", "imp_count_sp", "imp_count_pe", "steps", "epochs", "sec",
            "vram_mb", "smoke")}))
    journal.append_reflection(loop, it, status, f"{score:.4f}", f"{d:+.4f}", fields)
    distiller.update_backlog(loop, it)
    return {"iter": it, "status": status, "era_score": score, "best": new_best}


def run_loop(loop: Path, scorer, agent, *, max_iters: int = 10**9, stop_flag: Path | None = None) -> None:
    loop = Path(loop)
    # max_iters is a TOTAL target across sessions: seed `done` from productive iters already in
    # results.tsv so re-running start_menu after the ~11h login-node kill CONTINUES toward it (instead
    # of doing max_iters MORE). Rate-limit/skip waits never count.
    done = sum(1 for r in journal.read_results(loop) if r.get("status") in ("keep", "reset", "crash"))
    consec_ctx = consec_noedit = 0
    while done < max_iters:                        # NEVER STOP: driver never self-halts
        if stop_flag and Path(stop_flag).exists():
            break
        r = run_iteration(loop, scorer, agent)
        st = r.get("status")

        if st == "rate_limited":
            # both providers rate/usage-limited → WAIT for the limit to reset, then retry. No GPU
            # spent, no iteration burned. (A re-run of start_menu after the ~11h kill also resumes.)
            (loop / "RATE_LIMITED.flag").write_text(
                f"both agents rate/usage-limited; rechecking every {RATE_LIMIT_RECHECK_S}s "
                f"until the limit resets @ {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
            time.sleep(RATE_LIMIT_RECHECK_S)
            continue
        (loop / "RATE_LIMITED.flag").unlink(missing_ok=True)

        if st == "no_edit":                        # plain no-edit (NOT a detected rate limit)
            consec_noedit += 1
            if consec_noedit >= 5:
                (loop / "NEEDS_ATTENTION").write_text("5x consecutive no-edit (not rate-limited)\n")
                (loop / "DONE").write_text("halted on persistent no-edit\n")   # watchdog won't restart
                break
            time.sleep(120)
            continue
        consec_noedit = 0

        if st == "ctx_error":
            consec_ctx += 1
            if consec_ctx >= 5:                    # unrecoverable corruption: stop, flag the human
                (loop / "NEEDS_ATTENTION").write_text(f"ctx_error x{consec_ctx}: {r.get('problems')}\n")
                (loop / "DONE").write_text("halted on persistent ctx_error\n")   # watchdog won't restart
                break
            time.sleep(min(60, 10 * consec_ctx))  # backoff instead of spinning git subprocesses
            continue
        consec_ctx = 0

        done += 1                                  # only keep/reset/crash (a real scored attempt) counts
