"""Stateless step-N context contract (DESIGN_DOC §7).

`assemble(loop)` reconstructs the curated context bundle from the loop's durable memory
(program.md + best train.py + results.tsv + git history + last-K reflections + distilled backlog
+ run.log tail). `validate(loop, bundle)` runs the §7.2 assertions BEFORE any agent call; the
driver rebuilds/aborts on failure rather than spending a token on a stale or leaky context.

assemble() is PURE w.r.t. the on-disk artifacts (two calls → identical bundle) so it never races
the dashboard reader.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import journal

K_REFLECT = 8
TOKEN_CAP = 24000           # ~ chars/4; assemble trims older results rows into the backlog if over
BEST_REF = "best"           # git tag marking the champion commit


def _git(loop: Path, *args, ok_codes=(0,)) -> tuple[int, str]:
    r = subprocess.run(["git", "-C", str(loop), *args], capture_output=True, text=True)
    return r.returncode, (r.stdout if r.returncode in ok_codes else r.stdout + r.stderr)


def _toks(s: str) -> int:
    return len(s) // 4


# --------------------------------------------------------------------------- assemble
def assemble(loop: str | Path, k: int = K_REFLECT, token_cap: int = TOKEN_CAP) -> dict:
    loop = Path(loop).resolve()
    rows = journal.read_results(loop)
    n_committed = len(rows)
    next_iter = (max((int(r["iter"]) for r in rows), default=-1) + 1)

    program_md = (loop / "program.md").read_text() if (loop / "program.md").exists() else ""
    working_train = (loop / "train.py").read_text() if (loop / "train.py").exists() else ""
    rc, best_train = _git(loop, "show", f"{BEST_REF}:train.py")
    if rc != 0:
        best_train = working_train               # no best tag yet (pre-base) → working file
    best = journal.best_row(loop)
    best_score = best["era_score"] if best else "—"

    _, git_log = _git(loop, "log", "--oneline", "--decorate", "-n", "40")
    refl = journal.last_reflections(loop, k)
    bk_iter, backlog = journal.read_backlog(loop)
    run_log = ""
    if (loop / "run.log").exists():
        run_log = "\n".join((loop / "run.log").read_text().splitlines()[-25:])

    results_tsv = journal.results_path(loop).read_text() if journal.results_path(loop).exists() else ""

    def render(results_text: str, gitlog_text: str) -> str:
        refl_txt = "\n\n".join(
            f"## iter {b['iter']} · {b['status']} · era_score {b['era_score']} (Δbest {b['d_best']})\n"
            + "\n".join(f"- {f}: {b[f]}" for f in journal.REFLECTION_FIELDS) for b in refl) or "(none yet)"
        return (
            f"{program_md}\n\n"
            f"================ CURRENT BEST (the file to beat) — era_score {best_score} ================\n"
            f"```python\n{best_train}\n```\n\n"
            f"================ RESULTS so far (results.tsv) ================\n{results_text}\n"
            f"================ GIT HISTORY of THIS loop (inspect exact diffs with "
            f"`git show <sha>` / `git diff <a> <b>` — your branch only) ================\n{gitlog_text}\n"
            f"================ RECENT REFLECTIONS (last {k}) ================\n{refl_txt}\n\n"
            f"================ BACKLOG (distilled lessons + parked ideas) ================\n{backlog}\n"
            f"================ LAST RUN LOG (tail) ================\n{run_log or '(none)'}\n\n"
            f"================ YOUR TASK (iter {next_iter}) ================\n"
            f"Propose ONE surgical change along THIS loop's thesis (above) — edit ./train.py and/or the "
            f"vendored ./candi_model/ (your thesis says WHERE). Keep it minimal and attributable. "
            f"End by writing your reasoning to be journaled.\n")

    prompt = render(results_tsv, git_log)
    trimmed_older = 0
    # token-budget: drop OLDEST results rows (keep header + recent) into a note; reflections+backlog kept verbatim
    if _toks(prompt) > token_cap and results_tsv:
        lines = results_tsv.splitlines()
        header, body = lines[0], lines[1:]
        keep = body
        while _toks(prompt) > token_cap and len(keep) > k:
            trimmed_older = len(body) - (len(keep) - 1)
            keep = keep[1:]
            rt = header + "\n" + f"# (… {trimmed_older} older iterations compressed into BACKLOG) …\n" + "\n".join(keep)
            prompt = render(rt, git_log)

    return {
        "loop": str(loop), "next_iter": next_iter, "n_committed": n_committed,
        "best_score": best_score, "best_train": best_train, "working_train": working_train,
        "results_tsv": results_tsv, "git_log": git_log, "reflections": refl,
        "backlog_iter": bk_iter, "backlog": backlog, "prompt": prompt,
        "tokens": _toks(prompt), "trimmed_older": trimmed_older,
    }


# --------------------------------------------------------------------------- validate (§7.2)
def validate(loop: str | Path, bundle: dict, k: int = K_REFLECT, token_cap: int = TOKEN_CAP) -> tuple[bool, list[str]]:
    loop = Path(loop).resolve()
    problems: list[str] = []
    rows = journal.read_results(loop)
    n = len(rows)
    n_nonbase = sum(1 for r in rows if r.get("status") != "base")

    # [completeness]
    if not bundle["prompt"].strip():
        problems.append("completeness: empty prompt")
    if n >= 1 and not bundle["results_tsv"].strip():
        problems.append("completeness: results.tsv empty though iterations committed")
    if not bundle["working_train"].strip():
        problems.append("completeness: working train.py empty")

    # [provenance / freshness]
    rc, best_train = _git(loop, "show", f"{BEST_REF}:train.py")
    if rc == 0 and bundle["working_train"] != best_train:
        problems.append("provenance: working train.py != git show best:train.py (not at champion)")
    if n_nonbase > 0:                                  # backlog must have been distilled after the last iter
        if bundle["backlog_iter"] != (bundle["next_iter"] - 1):
            problems.append(f"provenance: backlog updated_at_iter={bundle['backlog_iter']} "
                            f"!= last completed iter {bundle['next_iter'] - 1}")
    if len(journal.read_reflections(loop)) != n_nonbase:
        problems.append(f"provenance: reflections count {len(journal.read_reflections(loop))} != "
                        f"non-base iters {n_nonbase}")

    # [schema]
    rp = journal.results_path(loop)
    if rp.exists():
        header = rp.read_text().splitlines()[0].split("\t")
        if header != journal.RESULTS_COLS:
            problems.append("schema: results.tsv header mismatch")
        for r in rows:
            if any(c not in r for c in journal.RESULTS_COLS):
                problems.append(f"schema: results row iter={r.get('iter')} missing columns"); break
            if r.get("status") not in journal.STATUSES:
                problems.append(f"schema: bad status {r.get('status')!r}"); break
    for b in bundle["reflections"]:
        if any(f not in b for f in journal.REFLECTION_FIELDS):
            problems.append(f"schema: reflection iter={b.get('iter')} missing fields"); break

    # [git access live]
    rc_log, log = _git(loop, "log", "--oneline")
    if rc_log != 0 or not log.strip():
        problems.append("git: log empty / unavailable")
    if rc != 0 and n >= 1:
        problems.append("git: `git show best:train.py` failed (no champion ref)")

    # [isolation / no-leakage] — bundle must not read from siblings or the frozen judge
    menu = loop.parent
    leaks = []
    for sib in menu.iterdir():
        if sib.is_dir() and sib != loop and (sib.name in ("_judge",) or not sib.name.startswith("_")):
            if str(sib.resolve()) in bundle["prompt"]:
                leaks.append(sib.name)
    if leaks:
        problems.append(f"leakage: bundle references foreign dirs {leaks}")

    # [token budget]
    if bundle["tokens"] > token_cap:
        problems.append(f"budget: {bundle['tokens']} tok > cap {token_cap} (distillation failed to fit)")
    # last-K reflections must survive verbatim regardless of trimming
    for b in bundle["reflections"]:
        if f"## iter {b['iter']} ·" not in bundle["prompt"]:
            problems.append(f"budget: last-K reflection iter={b['iter']} dropped from prompt"); break

    return (len(problems) == 0, problems)
