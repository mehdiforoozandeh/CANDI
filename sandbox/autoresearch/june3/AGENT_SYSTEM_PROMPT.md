# E34 june3 — mandatory agent system prompt

Paste this block at the **start** of every Cursor Agent session on branch `autoresearch/june3`.

---

## HARD SCOPE (non-negotiable)

You may **only** create or modify files under:

`sandbox/autoresearch/june3/`

**except** these frozen harness files (read-only):

- `prepare.py`, `pins.py`, `ar_fixed.yaml`, `pin_manifest.json`
- `validate_parity.py`, `validate_data_frac.py`, `eval_bridge.py`, `scope.py`, `agent_step.py`, `keep_rule.py`
- `program.md`, `README.md`, `AGENT_SYSTEM_PROMPT.md`, `loop.sh`, `ar_loop.py`, `hooks/pre-commit`

## ABSOLUTELY FORBIDDEN — file scope

- `sandbox/train.py` — **never** edit, patch, or “fix” eval signatures here
- `sandbox/candi_v2/` (production tree)
- `sandbox/train_candi_v2.py`, `sandbox/eval.py`, `sandbox/data.py`, `sandbox/batch.py`
- Any path outside `sandbox/autoresearch/june3/`

If `run_eval_pass` needs different kwargs, change **`june3/candi_v2/`** or **`june3/train.py`** only. Eval compatibility is handled by `june3/eval_bridge.py` (frozen).

## ABSOLUTELY FORBIDDEN — training config (enforced at runtime)

Even inside `get_config()`, you **must not** change any of these fields.
`prepare.py` validates them before training starts and will abort with an error if
any deviate from their `ar_fixed.yaml` values:

| Field | Frozen value |
|---|---|
| `training.optimizer.name` | `adamax` |
| `training.optimizer.adamax.lr` | `1e-3` |
| `training.grad.clip_norm` | `2.0` |
| `training.schedule.warmup_frac` | `0.1` |
| `training.loss_weights.obs_weight` | `3.5` |
| `training.loss_weights.imp_weight` | `0.59` |
| `training.loss_weights.count_weight` | `2.0` |
| `training.dsf.sampling` | `”off”` |
| `training.masking.*` | all frozen |
| `training.batch_size` | `4` |
| `training.amp` | `False` |

**Why:** these settings define the training conditions. Changing them makes
experiments incomparable — the gain would be from better training, not better
architecture. (Karpathy autoresearch principle: freeze *how* experiments run,
liberate *what* gets tested.)

**To add a new loss weight** (e.g. `kl_weight`): add a new field to
`june3/candi_v2/config.py`, NOT to `training.loss_weights`. See `SEARCH_SPACE.md`.

## Search space reference

Read `SEARCH_SPACE.md` before proposing hypotheses. It documents six tiers of
change from simple config switches (Tier 1, ~15 runs/hour) to full architectural
rearchitecting (Tier 5, ~5 runs/hour), with concrete code examples for each.

## Before every commit

```bash
python -m sandbox.autoresearch.june3.scope --staged
git add sandbox/autoresearch/june3/
git commit -m “autoresearch/june3: <desc>”
```

## Before every train

```bash
python -m sandbox.autoresearch.june3.agent_step --description “<desc>”
```

`agent_step` **refuses to run** if the worktree touches forbidden paths.
`prepare.py` **aborts training** if any frozen training config field is changed.

## Typical edit targets

- `sandbox/autoresearch/june3/candi_v2/encoder.py`, `decoder.py`, `model.py`, `loss.py`
- `sandbox/autoresearch/june3/candi_v2/config.py` — add new architecture config fields
- `sandbox/autoresearch/june3/train.py` — `get_config()` / `build_model()` only

Violating scope or frozen training config invalidates the experiment.
