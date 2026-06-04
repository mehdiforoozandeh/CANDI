#!/usr/bin/env python3
"""Unattended E34 june3 loop: apply queued get_config patches, train, keep/reset."""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
JUNE3 = Path(__file__).resolve().parent
TRAIN_PY = JUNE3 / "train.py"
NOTES = JUNE3 / "agent_notes.txt"
LOOP_START = JUNE3 / "loop_start_epoch.txt"
RUN_LOG = JUNE3 / "run.log"
DURATION_S = 5 * 3600

BASE_GET_CONFIG = '''def get_config() -> CANDIv2Config:
    """Return config for this run. Agent may edit encoder/decoder fields here."""
    return _load_baseline_config()'''

PATCHED_TEMPLATE = '''def get_config() -> CANDIv2Config:
    """Return config for this run. Agent may edit encoder/decoder fields here."""
    cfg = _load_baseline_config()
{body}
    return cfg'''

# (description, lines inside get_config after cfg = _load_baseline_config())
HYPOTHESES: list[tuple[str, str]] = [
    ("gated fusion + fusion_norm layer", "    cfg.encoder.fusion_mode = \"gated\"\n    cfg.encoder.fusion_norm = \"layer\""),
    ("metadata_embed_dim 64", "    cfg.encoder.metadata_embed_dim = 64\n    cfg.decoder.meta_embed_dim = 64"),
    ("nhead 8", "    cfg.encoder.nhead = 8"),
    ("dropout 0.05", "    cfg.encoder.dropout = 0.05"),
    ("decoder per_deconv_layer film", "    cfg.decoder.film_mode = \"per_deconv_layer\""),
    ("encoder film per_conv only", "    cfg.encoder.film_mode = \"per_conv\""),
    ("encoder expansion_factor 3", "    cfg.encoder.expansion_factor = 3"),
    ("encoder n_cnn_layers 4", "    cfg.encoder.n_cnn_layers = 4"),
    ("dna_pool_order early", "    cfg.encoder.dna_pool_order = \"early\""),
    ("decoder meta_embed_layernorm true", "    cfg.decoder.meta_embed_layernorm = True"),
    ("gated fusion + dropout 0.05", "    cfg.encoder.fusion_mode = \"gated\"\n    cfg.encoder.dropout = 0.05"),
    ("n_transformer_layers 3 + dropout 0.05", "    cfg.encoder.n_transformer_layers = 3\n    cfg.encoder.dropout = 0.05"),
    ("encoder conv_norm group", "    cfg.encoder.conv_norm = \"group\""),
    ("n_transformer_layers 3 + gated", "    cfg.encoder.n_transformer_layers = 3\n    cfg.encoder.fusion_mode = \"gated\""),
    ("metadata_embed_dim 48 + nhead 8", "    cfg.encoder.metadata_embed_dim = 48\n    cfg.encoder.nhead = 8\n    cfg.decoder.meta_embed_dim = 48"),
    ("encoder n_transformer_layers 3", "    cfg.encoder.n_transformer_layers = 3"),
    ("decoder expansion_factor 3", "    cfg.decoder.expansion_factor = 3"),
    ("encoder missing_data_mode mask_stem", "    cfg.encoder.missing_data_mode = \"mask_stem\""),
    ("production_dual transformer", "    cfg.encoder.transformer_type = \"production_dual\""),
    ("gated + n_transformer 3 + meta 48", "    cfg.encoder.fusion_mode = \"gated\"\n    cfg.encoder.n_transformer_layers = 3\n    cfg.encoder.metadata_embed_dim = 48\n    cfg.decoder.meta_embed_dim = 48"),
]


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(REPO), text=True, capture_output=True, **kw)


def _patch_train(body: str) -> None:
    text = TRAIN_PY.read_text()
    new_fn = PATCHED_TEMPLATE.format(body=body)
    text = re.sub(
        r"def get_config\(\)[^#]*?(?:return _load_baseline_config\(\)|return cfg)",
        new_fn,
        text,
        count=1,
        flags=re.DOTALL,
    )
    TRAIN_PY.write_text(text)


def _restore_baseline_train() -> None:
    text = TRAIN_PY.read_text()
    text = re.sub(
        r"def get_config\(\)[^#]*?(?:return _load_baseline_config\(\)|return cfg)",
        BASE_GET_CONFIG,
        text,
        count=1,
        flags=re.DOTALL,
    )
    TRAIN_PY.write_text(text)


def _parse_footer() -> dict[str, str]:
    if not RUN_LOG.exists():
        return {}
    lines = RUN_LOG.read_text().splitlines()
    out: dict[str, str] = {}
    in_block = False
    for line in lines:
        if line.strip() == "---":
            in_block = not in_block
            continue
        if in_block and ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def _append_note(commit: str, primary: str, verdict: str, lesson: str) -> None:
    with NOTES.open("a") as f:
        f.write(f"{commit} {primary} {verdict} {lesson}\n")


def _time_left() -> float:
    if not LOOP_START.exists():
        return DURATION_S
    t0 = float(LOOP_START.read_text().strip())
    return max(0.0, DURATION_S - (time.time() - t0))


def _slurm_alive() -> bool:
    r = _run(["bash", "-lc", "squeue -u $USER -n bash -h -o %T 2>/dev/null | head -1"])
    if not r.stdout.strip():
        return True
    return r.stdout.strip().upper() == "R"


def one_iteration(desc: str, body: str) -> None:
    _restore_baseline_train()
    _patch_train(body)
    if _run([sys.executable, "-m", "sandbox.autoresearch.june3.scope", "--staged"]).returncode != 0:
        _run(["git", "checkout", "--", "sandbox/autoresearch/june3/train.py"])
        return
    _run(["git", "add", "sandbox/autoresearch/june3/"])
    commit_msg = f"autoresearch/june3: {desc}"
    cr = _run(["git", "commit", "-m", commit_msg])
    if cr.returncode != 0:
        return
    commit = _run(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(REPO)}
    proc = subprocess.run(
        [sys.executable, "-m", "sandbox.autoresearch.june3.agent_step", "--description", desc],
        cwd=str(REPO),
        env=env,
    )
    footer = _parse_footer()
    primary = footer.get("primary_score", "nan")
    verdict_line = "keep"
    if "KEEP_VERDICT: keep" not in (RUN_LOG.read_text() if RUN_LOG.exists() else ""):
        # agent_step prints to stdout; re-check results tsv last line
        lines = (JUNE3 / "results.tsv").read_text().splitlines()
        if len(lines) > 1:
            parts = lines[-1].split("\t")
            if len(parts) > 1:
                verdict_line = parts[1]
    if verdict_line != "keep":
        _run(["git", "reset", "--hard", "HEAD~1"])
        _restore_baseline_train()
    lesson = desc[:80]
    _append_note(commit, primary, verdict_line, lesson)


def main() -> int:
    idx = 0
    while _time_left() > 300 and _slurm_alive():
        desc, body = HYPOTHESES[idx % len(HYPOTHESES)]
        idx += 1
        print(f"[ar_loop] {desc} (left {_time_left()/3600:.2f}h)", flush=True)
        try:
            one_iteration(desc, body)
        except Exception as exc:
            print(f"[ar_loop] crash: {exc}", flush=True)
            _run(["git", "reset", "--hard", "HEAD~1"])
            _restore_baseline_train()
        time.sleep(2)
    print("[ar_loop] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
