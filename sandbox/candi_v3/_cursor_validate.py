"""Verify cursor-agent (composer) can serve as the ERA generator: run the REAL generate
prompt through the production cursor path and confirm it returns a valid program that passes
the CPU smoke test. Uses the best node (161 / cand0162) as the parent."""
import sys
import time

sys.path.insert(0, ".")
from futs import Solution                                 # noqa: E402
from generate import Generator, extract_program, make_prompt   # noqa: E402
import problem                                            # noqa: E402

prob, _seed, _ = problem.get_problem()
parent = open("runs/cand0162/program.py").read()          # best genuine node (161)
prompt = make_prompt(prob, Solution(parent), -0.003621)

g = Generator(backend="cursor", cursor_model="composer-2.5-fast", timeout_s=600)
print("[cursor-validate] cmd:", g._cmd("cursor"))
print("[cursor-validate] generating (composer)...", flush=True)
t0 = time.time()
out, limited, ok = g._run("cursor", prompt)
dt = time.time() - t0

prog = extract_program(out)
valid = bool(prog and "class Model" in prog and "class Objective" in prog and "run_and_score" in prog)
print(f"[cursor-validate] elapsed={dt:.0f}s  returncode_ok={ok}  limited={limited}")
print(f"[cursor-validate] valid program returned: {valid}  (len={len(prog)})")

if valid:
    sok, err = g._smoke(prog)
    print(f"[cursor-validate] SMOKE: {'PASS' if sok else 'FAIL'}")
    if not sok:
        print(err[-1500:])
    open("_cursor_validate_prog.py", "w").write(prog)
else:
    print("[cursor-validate] no valid program. First 800 chars of cursor output:")
    print(out[:800])
