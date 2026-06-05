# E34 architecture autoresearch (june3)

Karpathy loop on **vendored CANDI v2** with frozen v2 train/eval.

Spec: [`sandbox/ideas/autoresearch_june3_arch.md`](../../ideas/autoresearch_june3_arch.md)

```bash
python -m sandbox.autoresearch.june3.validate_parity
python -m sandbox.autoresearch.june3.train
python -m sandbox.autoresearch.june3.agent_step --description "..."
python -m sandbox.autoresearch.june3.scope --staged

Guards: `imp_count_r2_gw > 0`, `den_count_r2_gw > 0`, DCR ∈ [3, 5], params ≤ 5× baseline, VRAM ≤ 9500 MB.

Overnight: see `program.md` + `AGENT_SYSTEM_PROMPT.md`. Install optional git hook:

`ln -sf ../../sandbox/autoresearch/june3/hooks/pre-commit .git/hooks/pre-commit`

`agent_step` refuses to run if anything outside `june3/` is modified (e.g. `sandbox/train.py`).
```

Agent playbook: [`program.md`](program.md)
