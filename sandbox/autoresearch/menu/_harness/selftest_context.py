"""G4 — adversarial selftest for the stateless context contract (context.py / journal / distiller).

Builds a synthetic loop (nested git repo, base + iterations), asserts a clean bundle VALIDATES,
then asserts the validator REJECTS each single corruption. A validator that never rejects is
worthless — every adversarial case must produce a problem.

Run:  python _harness/selftest_context.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import journal      # noqa: E402
import distiller    # noqa: E402
import context      # noqa: E402


def _git(d, *a):
    subprocess.run(["git", "-C", str(d), *a], check=True, capture_output=True, text=True)


def _commit(loop, msg):
    _git(loop, "add", "-A")
    _git(loop, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", msg)


ASPECTS = dict(S_A="0.0", Q_imp="0.4857", Q_den="0.50", ece="0.071", dcr="4.0", c_index="0.50",
               peak_auroc="0.72", imp_pval_sp="0.55", imp_pval_pe="0.34", imp_count_sp="0.53",
               imp_count_pe="0.30", steps="1200", epochs="4.0", sec="1790", vram_mb="3800", smoke="pass")


def build_loop(root: Path, n_iters: int = 10) -> Path:
    loop = root / "single_lambda"
    loop.mkdir(parents=True)
    (loop / "program.md").write_text("# THESIS: single latent enrichment field. Edit train.py only.\n")
    (loop / "train.py").write_text("# row0 candi_v2\nVERSION = 0\n")
    _git(loop, "init", "-q")
    _commit(loop, "iter 0 base")
    _git(loop, "tag", "-f", context.BEST_REF)
    journal.append_result(loop, dict(iter=0, ts="t0", commit="base", parent="-", status="base",
                                     era_score="0.012", d_best="-", change_summary="candi_v2 as-is (row-0)", **ASPECTS))
    distiller.update_backlog(loop, 0)

    best = 0.012
    for it in range(1, n_iters + 1):
        score = round(0.012 + 0.002 * it, 4)
        keep = score > best
        (loop / "train.py").write_text(f"# row0 candi_v2\nVERSION = {it}\n# change {it}\n")
        _commit(loop, f"iter {it}")
        if keep:
            _git(loop, "tag", "-f", context.BEST_REF)
            best = score
        else:                                   # reject → restore champion working file
            _git(loop, "checkout", context.BEST_REF, "--", "train.py")
        d = round(score - best, 4)
        journal.append_result(loop, dict(iter=it, ts=f"t{it}", commit=f"c{it}", parent=f"c{it-1}",
                                          status="keep" if keep else "reset", era_score=str(score),
                                          d_best=str(d), change_summary=f"change {it}", **ASPECTS))
        journal.append_reflection(loop, it, "keep" if keep else "reset", score, d, dict(
            hypothesis=f"try change {it}", rationale="per thesis", expected="lift Q_imp",
            result=f"scored {score}", interpretation="kept" if keep else "reset",
            parked=f"idea {it} for later"))
        distiller.update_backlog(loop, it)
    return loop


def expect(cond, msg):
    assert cond, "FAIL: " + msg
    print(f"  PASS {msg}")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="g4_ctx_"))
    try:
        loop = build_loop(tmp, n_iters=60)

        # clean bundle validates
        b = context.assemble(loop)
        ok, probs = context.validate(loop, b)
        expect(ok, f"clean bundle validates ({probs})")

        # git-diff affordance is live
        rc, best_src = context._git(loop, "show", f"{context.BEST_REF}:train.py")
        expect(rc == 0 and "VERSION" in best_src, "git show best:train.py works (exact diff access)")
        expect("git show" in b["prompt"] and "git diff" in b["prompt"], "prompt advertises git diff access")

        # determinism
        expect(context.assemble(loop)["prompt"] == b["prompt"], "assemble is deterministic")

        # --- adversarial: each must make validate REJECT ---
        def reject(tag):
            ok2, p2 = context.validate(loop, context.assemble(loop))
            expect(not ok2, f"REJECTS {tag} ({p2})")

        (loop / "train.py").write_text("# tampered working file\n"); reject("stale working train.py")
        context._git(loop, "checkout", context.BEST_REF, "--", "train.py")    # restore

        refl = journal.reflections_path(loop)
        saved = refl.read_text()
        refl.write_text(saved.rsplit("## iter", 1)[0])                        # drop last reflection
        reject("missing reflection")
        refl.write_text(saved)                                                 # restore

        bk = journal.backlog_path(loop); bksaved = bk.read_text()
        cur_iter, _ = journal.read_backlog(loop)
        bk.write_text(bksaved.replace(f"updated_at_iter: {cur_iter}", "updated_at_iter: 3"))
        reject("stale backlog")
        bk.write_text(bksaved)

        rp = journal.results_path(loop); rpsaved = rp.read_text()
        rp.write_text("iter\tBOGUS\n" + "\n".join(rpsaved.splitlines()[1:]))
        reject("corrupt results header")
        rp.write_text(rpsaved)

        prog = loop / "program.md"; progsaved = prog.read_text()
        sib = tmp / "axial_longrange"; sib.mkdir()                            # a sibling loop
        prog.write_text(progsaved + f"\nLEAK {sib.resolve()}/train.py\n")
        reject("injected sibling-loop path (leakage)")
        prog.write_text(progsaved)

        # clean again after restores
        ok3, p3 = context.validate(loop, context.assemble(loop))
        expect(ok3, f"re-validates clean after restores ({p3})")

        # over-cap history → assemble trims OLDER results rows; pick a cap between floor and full
        full = context.assemble(loop, token_cap=10**9)
        floor = context.assemble(loop, token_cap=1)            # maximally trimmed (down to last-K rows)
        cap = (floor["tokens"] + full["tokens"]) // 2
        big = context.assemble(loop, token_cap=cap)
        expect(full["tokens"] > cap > floor["tokens"], "test cap sits between floor and full")
        expect(big["trimmed_older"] > 0 and "compressed into BACKLOG" in big["prompt"], "over-cap trims older results")
        okb, pb = context.validate(loop, big, token_cap=cap)
        expect(okb, f"trimmed bundle fits & validates ({pb})")
        last_k = journal.last_reflections(loop, context.K_REFLECT)
        expect(all(f"## iter {r['iter']} ·" in big["prompt"] for r in last_k), "last-K reflections preserved verbatim under trim")

        # a cap below the irreducible floor must be FLAGGED, never silently truncated
        okf, pf = context.validate(loop, context.assemble(loop, token_cap=50), token_cap=50)
        expect(not okf and any("budget" in x for x in pf), "un-fittable cap is flagged, not silently truncated")

        print("G4 CONTEXT-CONTRACT: ALL PASS")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
