"""G4 — observed 3-iteration ratchet run with mock agent/scorer/smoke (fast, no GPU).

Validates the DRIVER plumbing the real GPU scorer + claude agent will use: keep/reset/Δ correct,
every attempt preserved in git history (exact diffs), `best` tag tracks the champion, working
train.py restored on reset, journals + distiller updated, and the NEXT context re-validates each
iteration. The real scorer is validated separately by G1.

Run:  python _harness/selftest_driver.py
"""
from __future__ import annotations

import re
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import context      # noqa: E402
import driver       # noqa: E402
import journal      # noqa: E402

# planned hints per iter: keep(0.020) → reset(0.018<best) → keep(0.025)
PLAN = {1: 0.020, 2: 0.018, 3: 0.025}
EXPECT = {1: "keep", 2: "reset", 3: "keep"}
EXPECT_BEST = {1: 0.020, 2: 0.020, 3: 0.025}
FIXED = dict(S_A="0.0", Q_imp="0.486", Q_den="0.50", ece="0.071", dcr="4.0", c_index="0.50",
             peak_auroc="0.72", imp_pval_sp="0.55", imp_pval_pe="0.34", imp_count_sp="0.53",
             imp_count_pe="0.30", steps="1200", epochs="4.0", sec="1790", vram_mb="3800")


def mock_agent(loop, bundle):
    it = bundle["next_iter"]
    txt = (loop / "train.py").read_text()
    hint = PLAN[it]
    txt = re.sub(r"SCORE_HINT = [\d.]+", f"SCORE_HINT = {hint}", txt)
    txt += f"\n# edit iter {it}\n"
    (loop / "train.py").write_text(txt)
    return (f"set hint {hint} (iter {it})",
            dict(hypothesis=f"change {it}", rationale="thesis", expected="lift Q_imp",
                 parked=f"idea {it}"))


def mock_scorer(loop, train_py):
    m = re.search(r"SCORE_HINT = ([\d.]+)", Path(train_py).read_text())
    return float(m.group(1)), dict(FIXED), "ok"


def mock_smoke(loop):
    return True


def expect(c, m):
    assert c, "FAIL: " + m
    print(f"  PASS {m}")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="g4_drv_"))
    try:
        loop = tmp / "single_lambda"
        row0 = tmp / "row0.py"
        row0.write_text("# row0\nSCORE_HINT = 0.012\n")
        (tmp / "candi_model").mkdir()                       # init_loop vendors an editable model pkg
        (tmp / "candi_model" / "model.py").write_text("# vendored model fork\n")
        driver.init_loop(loop, row0, "# THESIS single_lambda. Edit train.py only.\n", 0.012, dict(FIXED))

        ok, p = context.validate(loop, context.assemble(loop))
        expect(ok, f"context valid after base ({p})")

        for it in (1, 2, 3):
            r = driver.run_iteration(loop, mock_scorer, mock_agent, smoke_fn=mock_smoke)
            expect(r["status"] == EXPECT[it], f"iter {it} status={r['status']} == {EXPECT[it]}")
            expect(abs(r["best"] - EXPECT_BEST[it]) < 1e-9, f"iter {it} best={r['best']} == {EXPECT_BEST[it]}")
            ok, p = context.validate(loop, context.assemble(loop))
            expect(ok, f"context re-validates after iter {it} ({p})")

        rows = journal.read_results(loop)
        expect([r["status"] for r in rows] == ["base", "keep", "reset", "keep"], "results status sequence")

        # every attempt preserved in git history (exact diffs), incl. the rejected iter 2
        log = driver._git(loop, "log", "--oneline").stdout.strip().splitlines()
        expect(len(log) == 4, f"git history has all 4 attempts (got {len(log)})")
        expect(any("iter 2" in l for l in log), "rejected iter-2 commit preserved in history (diff inspectable)")

        # champion == iter-3 version; working train.py restored to champion
        best_src = driver._git(loop, "show", f"{context.BEST_REF}:train.py").stdout
        expect("SCORE_HINT = 0.025" in best_src, "best tag == iter-3 champion")
        expect((loop / "train.py").read_text() == best_src, "working train.py == champion (provenance)")

        # reset (iter 2) restored champion's working file at the time (hint 0.020 from iter 1)
        refl = journal.read_reflections(loop)
        expect(len(refl) == 3 and refl[1]["iter"] == 2 and refl[1]["status"] == "reset", "iter-2 reflection journaled")
        expect(journal.read_backlog(loop)[0] == 3, "backlog distilled to iter 3")
        # the kept-lineage shows in backlog lessons; reset shows as rejected
        _, bk = journal.read_backlog(loop)
        expect("kept (validated lineage)" in bk and "rejected" in bk, "backlog separates kept vs rejected")

        print("G4 DRIVER (3-iter ratchet): ALL PASS")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
