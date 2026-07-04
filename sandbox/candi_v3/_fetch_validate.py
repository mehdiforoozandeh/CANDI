"""Live validation: run the REAL generate prompt (with §3.6 repo links) through the modified
claude headless path, confirm it (a) actually calls WebFetch, (b) returns a valid program,
(c) the program passes the CPU smoke test. One generation + one smoke."""
import json
import subprocess
import sys
import time

sys.path.insert(0, ".")
from futs import Solution                         # noqa: E402
from generate import Generator, extract_program, make_prompt   # noqa: E402
import problem                                    # noqa: E402

prob, _seed, _ = problem.get_problem()
parent_src = open("runs/cand0071/program.py").read()   # current best (node 70) as the parent
prompt = make_prompt(prob, Solution(parent_src), -0.031521)

cmd = ["claude", "-p", "--model", "opus", "--effort", "medium",
       "--allowedTools", "WebFetch", "--output-format", "stream-json", "--verbose"]
print("[validate] launching real generation (WebFetch enabled)...", flush=True)
t0 = time.time()
p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=540)
raw = p.stdout
open("_fetch_validate.out", "w").write(raw)

nfetch = raw.count('"name":"WebFetch"') + raw.count('"name": "WebFetch"')
fetched_urls = []
texts = []
for line in raw.splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except Exception:
        continue
    if obj.get("type") == "assistant":
        for blk in obj.get("message", {}).get("content", []):
            if blk.get("type") == "text":
                texts.append(blk["text"])
            if blk.get("type") == "tool_use" and blk.get("name") == "WebFetch":
                fetched_urls.append(blk.get("input", {}).get("url", "?"))
    if obj.get("type") == "result" and isinstance(obj.get("result"), str):
        texts.append(obj["result"])

prog = extract_program("\n".join(texts))
open("_fetch_validate_prog.py", "w").write(prog)
valid = bool(prog and "class Model" in prog and "run_and_score" in prog)

print(f"[validate] elapsed={time.time()-t0:.0f}s  returncode={p.returncode}")
print(f"[validate] WebFetch calls: {nfetch}  urls={fetched_urls}")
print(f"[validate] valid program returned: {valid}  (len={len(prog)})")

if valid:
    ok, err = Generator()._smoke(prog)
    print(f"[validate] SMOKE: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(err[-1500:])
else:
    print("[validate] no valid program; first 800 chars of model output:")
    print(("\n".join(texts))[:800])
