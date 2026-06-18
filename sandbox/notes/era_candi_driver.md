# CANDI driver for an ERA problem.py (salvaged from the old `candi-era` skill)

The general ERA skill now lives at `~/.claude/skills/era/` (domain-agnostic,
candidate = whole program). It no longer ships a CANDI baseline. This note keeps
the CANDI-specific glue so you can write a `problem.py` for a CANDI search
(v2 config-tuning *or* a future candi_v3 whole-program search) without re-deriving it.

## SLURM / env for `config.yaml`
```yaml
exec_mode: sbatch_per_run
python: /project/6014832/mforooz/EpiDenoise/candi_venv/bin/python
account: def-maxwl_gpu
gres: gpu:nvidia_h100_80gb_hbm3_1g.10gb:1   # MANDATORY exact spec (CLAUDE.md hard constraint)
cpus: 2
mem: 12G
time_limit: "0:30:00"
setup_cmds:
  - "source /home/mforooz/miniconda3/etc/profile.d/conda.sh 2>/dev/null"
  - "conda activate candi 2>/dev/null"
  - "source /project/6014832/mforooz/EpiDenoise/candi_venv/bin/activate 2>/dev/null"
  - "module load samtools 2>/dev/null"
  - "cd /project/6014832/mforooz/EpiDenoise"
  - "export PYTHONPATH=/project/6014832/mforooz/EpiDenoise"
```
(The general executor runs each candidate in its own temp workdir; add the
`cd` + `PYTHONPATH` lines so `sandbox` imports resolve.)

## Baseline program — drives the production trainer
The old config-only lever drove `sandbox.train_candi_v2` with the promoted
KEEP12 arch (`e34_v2_full_stack.yaml`) at the june3 budget, then scored from
`metrics.jsonl`. Baseline invocation:
```python
from sandbox import train_candi_v2
import tempfile
run_dir = tempfile.mkdtemp(prefix="era_run_")
argv = ["--config", "sandbox/configs/e34_v2_full_stack.yaml",
        "--epochs", "20", "--batch-size", "4",
        "--set", "eval.eval_every_n_epochs=1",
        "--no-wandb", "--run-dir", run_dir]
rc = train_candi_v2.main(argv)   # 0 on success
```
`candi_v2_default` supplies the regime (type1_chr19; AR loss weights obs 3.5 /
imp 0.59 / count 2.0; adamax; amp off; sandbox.h5). For a **whole-program candi_v3
search**, the candidate would instead import/construct the v3 model directly
(not via the v2 trainer) — but the env block above and the scoring fn below carry over.

## Scoring from metrics.jsonl (the june3 default objective)
```python
def score_from_metrics(path):
    """primary = 0.45*imp_count_r2_gw + 0.25*den_count_r2_gw
                 - 0.20*count_imp_loss - 0.10*count_obs_loss,
       at the best epoch (lowest total count loss). Non-finite term -> -1e9."""
    import json, math
    def get(d, name):
        if not isinstance(d, dict): return None
        for k, v in d.items():
            if k == name or k.endswith("/" + name): return v
        return None
    cands = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except Exception: continue
            em, el = r.get("eval_metrics"), r.get("eval_losses")
            vals = [get(em, "imp_count_r2_gw"), get(em, "den_count_r2_gw"),
                    get(el, "count_imp_loss"), get(el, "count_obs_loss")]
            if any(v is None or not math.isfinite(float(v)) for v in vals): continue
            imp_r2, den_r2, imp_loss, obs_loss = map(float, vals)
            primary = 0.45*imp_r2 + 0.25*den_r2 - 0.20*imp_loss - 0.10*obs_loss
            cands.append((imp_loss + obs_loss, primary))
    return min(cands, key=lambda c: c[0])[1] if cands else -1e9
```
metrics.jsonl 'epoch' records hold nested `eval_metrics` / `eval_losses` dicts
with slash-prefixed inner keys (e.g. `eval_metrics/imp_count_r2_gw`) — match by suffix.

> Per CRITIQUE.md §7: for candi_v3, select on the **downstream** objective
> (imputation R² ≫ obs, + a calibration term), not the obs-weighted recon loss.
> Re-design this scoring fn before a v3 search; lock it before searching.
