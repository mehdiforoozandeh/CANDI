"""The gates that guard the ingestion landmines, on a tiny synthetic ENCODE-layout fixture.

No real data: a 3-assay / 4-biosample / 2-chromosome directory tree and a 400 kb FASTA are built in
tmp_path, and a schema-v2 h5 is written by hand where only the dataset side is under test. Every assay
name here is synthetic (`assay0`...) — the kit must never depend on a real assay-name literal.

Each test names the landmine it guards; see .BUILD_PLAN FILE MANIFEST -> prep/handler.py (E3-E5),
prep/bake.py (B6, B8) and prep/reference_sample.py (the bijection gate).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

RES = 25
CTX_BINS = 384
CTX_BP = CTX_BINS * RES              # 9,600
CHROM_LEN = 200_000
N_BINS = CHROM_LEN // RES            # 8,000
TRAIN_CHROM, EVAL_CHROM = "chr20", "chr22"
ASSAYS = ("assay0", "assay1", "assay2")
BIOSAMPLES = ("T_a", "T_b", "V_a", "B_a")
DSFS = (1, 2, 4, 8)
DEPTH1 = 40_000_000
CONTROL_DIR = "chipseq-control"


# ---------------------------------------------------------------------------
# synthetic ENCODE-layout fixture
# ---------------------------------------------------------------------------

def _file_metadata(assay: str, *, run_type: str = "single-ended", read_length: int = 36) -> dict:
    # The handler reads these as one-entry dicts keyed by an arbitrary index string
    # (data.py:1238-1243 takes list(d.keys())[0]).
    return {
        "assay": {"1": assay},
        "accession": {"1": "ENCFF000TST"},
        "read_length": {"1": read_length},
        "run_type": {"1": run_type},
        "sequencing_platform": {"1": "Illumina NovaSeq 6000"},
        "lab": {"1": "test lab"},
        "biosample": {"1": "ENCBS000TST"},
    }


def _write_signal(dirpath: Path, dsf: int, values: dict, *, depth: int) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    for chrom, arr in values.items():
        np.savez(dirpath / f"{chrom}.npz", arr)
    (dirpath / "metadata.json").write_text(json.dumps({"depth": depth, "coverage": 0.5, "dsf": dsf}))


def _counts(seed: int, dsf: int, *, peak_value: int = 0) -> dict:
    out = {}
    for k, chrom in enumerate((TRAIN_CHROM, EVAL_CHROM)):
        rng = np.random.default_rng(seed * 10 + k)
        a = rng.poisson(8.0 / dsf, N_BINS).astype(np.int64)
        a[7] = 500 // dsf                      # a real peak: the bake's F7 gate wants max count > 50
        if peak_value:
            a[100] = peak_value
        out[chrom] = a
    return out


def make_root(root: Path, *, run_types: dict | None = None, huge_count_at: tuple | None = None,
              drop_dsf: dict | None = None) -> Path:
    """Build the ENCODE-style directory. Fault injectors: `run_types[(bios,assay)]` overrides the
    run_type string, `huge_count_at=(bios,assay)` puts a count past int16, `drop_dsf[bios]` omits a
    DSF level for every assay of that biosample."""
    run_types = run_types or {}
    huge = huge_count_at
    drop_dsf = drop_dsf or {}
    for bi, bios in enumerate(BIOSAMPLES):
        for ai, assay in enumerate(ASSAYS):
            d = root / bios / assay
            d.mkdir(parents=True, exist_ok=True)
            rt = run_types.get((bios, assay), "paired-ended" if ai % 2 else "single-ended")
            (d / "file_metadata.json").write_text(json.dumps(_file_metadata(assay, run_type=rt)))
            peak = 40_000 if huge == (bios, assay) else 0
            for dsf in DSFS:
                if dsf in drop_dsf.get(bios, ()):
                    continue
                _write_signal(d / f"signal_DSF{dsf}_res{RES}", dsf,
                              _counts(bi * 7 + ai, dsf, peak_value=peak), depth=DEPTH1 // dsf)
            pk = {c: (v > 12).astype(np.int64) for c, v in _counts(bi * 7 + ai, 1).items()}
            _write_signal(d / f"peaks_res{RES}", 1, pk, depth=DEPTH1)
            bw = {c: v.astype(np.float32) / 4.0 for c, v in _counts(bi * 7 + ai, 1).items()}
            _write_signal(d / f"signal_BW_res{RES}", 1, bw, depth=DEPTH1)
        c = root / bios / CONTROL_DIR
        c.mkdir(parents=True, exist_ok=True)
        (c / "file_metadata.json").write_text(json.dumps(_file_metadata(CONTROL_DIR)))
        for dsf in DSFS:
            _write_signal(c / f"signal_DSF{dsf}_res{RES}", dsf, _counts(100 + bi, dsf),
                          depth=DEPTH1 // dsf)
    return root


def make_side(tmp: Path):
    """chrom.sizes + a small indexed FASTA. Returns a validated SideFiles."""
    pysam = pytest.importorskip("pysam", reason="the bake needs the [prepare] extra")
    from candi_kit.prep.paths import SideFiles

    sizes = tmp / "test.chrom.sizes"
    sizes.write_text("".join(f"{c}\t{CHROM_LEN}\n" for c in (TRAIN_CHROM, EVAL_CHROM)))
    fa = tmp / "test.fa"
    rng = np.random.default_rng(0)
    with fa.open("w") as fh:
        for chrom in (TRAIN_CHROM, EVAL_CHROM):
            fh.write(f">{chrom}\n")
            seq = "".join(np.array(list("ACGT"))[rng.integers(0, 4, CHROM_LEN)])
            for i in range(0, CHROM_LEN, 60):
                fh.write(seq[i:i + 60] + "\n")
    pysam.faidx(str(fa))
    return SideFiles(chrom_sizes=sizes, fasta=fa)


def make_panel(assays=ASSAYS):
    from candi_kit.prep.panel import Panel
    return Panel(assays=tuple(assays), biosamples=tuple(BIOSAMPLES), dsf_list=DSFS,
                 resolution=RES, context_bins=CTX_BINS,
                 train_chroms=(TRAIN_CHROM,), eval_chroms=(EVAL_CHROM,))


@pytest.fixture(scope="module")
def good_bake(tmp_path_factory):
    """Bake the clean fixture once; every non-fault test reads this h5."""
    pytest.importorskip("pysam", reason="the bake needs the [prepare] extra")
    from candi_kit.prep.bake import bake

    tmp = tmp_path_factory.mktemp("good")
    root = make_root(tmp / "root")
    side = make_side(tmp)
    out = tmp / "out" / "good.h5"
    bake(root, make_panel(), out, side, max_tile_per_chrom=2)
    return out


# ---------------------------------------------------------------------------
# hand-built schema-v2 h5 (dataset side under test, bake not involved)
# ---------------------------------------------------------------------------

def write_v2_h5(path: Path, *, num_assays: int = 3, context_bins: int = CTX_BINS,
                n_train: int = 4, n_eval: int = 2, unavailable: tuple = ()) -> Path:
    assays = [f"assay{i}" for i in range(num_assays)]
    n, L, F = n_train + n_eval, context_bins, num_assays
    Lbp = L * RES
    rng = np.random.default_rng(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["version"] = 2
        h5.attrs["context_bins"] = L
        h5.attrs["resolution"] = RES
        h5.attrs["data_root"] = "synthetic"
        h5.attrs["kit_version"] = "test"
        h5.attrs["assays"] = json.dumps(assays)
        h5.attrs["assay_ids"] = json.dumps(list(range(F)))
        h5.attrs["requested_panel"] = json.dumps(assays)
        h5.attrs["num_assays"] = F
        h5.attrs["control_assay_id"] = F
        h5.attrs["dsf_list"] = json.dumps(list(DSFS))
        h5.attrs["train_chroms"] = json.dumps([TRAIN_CHROM])
        h5.attrs["eval_chroms"] = json.dumps([EVAL_CHROM])
        h5.attrs["panel_json"] = json.dumps({"assays": assays})

        chroms = [TRAIN_CHROM] * n_train + [EVAL_CHROM] * n_eval
        ws = h5.create_group("windows")
        ws.create_dataset("chrom", data=np.array(chroms, dtype=object),
                          dtype=h5py.string_dtype(encoding="utf-8"))
        ws.create_dataset("start", data=np.arange(n, dtype=np.int64) * L * RES)
        ws.create_dataset("end", data=(np.arange(n, dtype=np.int64) + 1) * L * RES)
        ws.create_dataset("region_type", data=np.full(n, 255, dtype=np.uint8))

        order = ["T_a", "V_a"]
        bg = h5.create_group("biosamples")
        bg.attrs["order"] = json.dumps(order)
        for bios in order:
            g = bg.create_group(bios)
            for dsf in DSFS:
                c = rng.integers(0, 201 // dsf, size=(n, L, F)).astype(np.int16)
                for fi in unavailable:
                    c[:, :, fi] = 0
                g.create_dataset(f"counts_dsf{dsf}", data=c)
                md = np.zeros((4, F), dtype=np.float32)
                md[0] = math.log2(DEPTH1 / dsf)
                md[1] = np.arange(F)
                md[2] = 36.0
                md[3] = 0.0
                for fi in unavailable:
                    md[:, fi] = -1.0
                g.create_dataset(f"meta_dsf{dsf}", data=md)
            g.create_dataset("pval", data=np.zeros((n, L, F), dtype=np.float16))
            g.create_dataset("peaks", data=np.zeros((n, L, F), dtype=np.int64))
            g.create_dataset("control", data=np.ones((n, L, 1), dtype=np.float32))
            cm = np.zeros((n, 4, 1), dtype=np.float32)
            cm[:, 0, 0] = math.log2(DEPTH1)
            cm[:, 1, 0] = F
            cm[:, 2, 0] = 36.0
            g.create_dataset("control_meta", data=cm)
            dna = np.zeros((n, Lbp, 4), dtype=np.int8)
            dna[:, :, 0] = 1
            g.create_dataset("dna", data=dna)
    return path


@pytest.fixture(scope="module")
def v2_h5(tmp_path_factory):
    return write_v2_h5(tmp_path_factory.mktemp("v2") / "v2.h5")


# ---------------------------------------------------------------------------
# (1) panel/alias bijection
# ---------------------------------------------------------------------------

def test_panel_alias_mismatch_names_the_missing_assay(tmp_path):
    """A panel assay absent from the resolved alias order must raise BY NAME, never permute silently.
    The realistic trigger is a stale aliases.json in the data root (it is written on the first handler
    build and never rewritten), which is exactly why it must never be deleted once a bake exists."""
    pytest.importorskip("pysam", reason="the bake needs the [prepare] extra")
    from candi_kit.prep.reference_sample import make_handler

    root = make_root(tmp_path / "root")
    side = make_side(tmp_path)
    make_handler(root, make_panel(), side)                    # writes aliases.json for ASSAYS
    with pytest.raises(ValueError) as ei:
        make_handler(root, make_panel(ASSAYS + ("ghost_assay",)), side)
    assert "ghost_assay" in str(ei.value)


def test_bijection_holds_on_the_clean_panel(good_bake):
    with h5py.File(good_bake, "r") as h5:
        assays = json.loads(h5.attrs["assays"])
        assert sorted(assays) == sorted(ASSAYS)
        assert json.loads(h5.attrs["assay_ids"]) == list(range(len(assays)))
        assert int(h5.attrs["control_assay_id"]) == len(assays)
        assert int(h5.attrs["version"]) == 2


# ---------------------------------------------------------------------------
# (2) an absent DSF level is -1, NEVER 0
# ---------------------------------------------------------------------------

def test_absent_dsf_level_writes_minus_one_meta(tmp_path):
    """B6, the worst latent bug in the pipeline: a 0-filled meta row marks EVERY assay available at
    log2(depth)=0 / assay_id=0 with all-zero counts, so training descends on garbage. The -1 sentinel
    is what makes the column read as unavailable. (_verify must skip -1 columns in the F4 check.)"""
    pytest.importorskip("pysam", reason="the bake needs the [prepare] extra")
    from candi_kit.prep.bake import bake

    root = make_root(tmp_path / "root", drop_dsf={"T_b": (8,)})
    side = make_side(tmp_path)
    out = tmp_path / "out" / "dropdsf.h5"
    bake(root, make_panel(), out, side, max_tile_per_chrom=2)
    with h5py.File(out, "r") as h5:
        md8 = np.array(h5["biosamples"]["T_b"]["meta_dsf8"])
        assert np.all(md8 == -1.0), f"absent DSF level must be -1-filled, got {md8}"
        assert np.all(np.array(h5["biosamples"]["T_a"]["meta_dsf8"])[0] > 0)


def test_dataset_treats_minus_one_meta_as_unavailable(tmp_path):
    from candi_kit.dataset import CandiKitH5Dataset

    p = write_v2_h5(tmp_path / "unavail.h5", unavailable=(1,))
    ds = CandiKitH5Dataset(p, "type1", train=True, batch_size=2, dsf_sampling="off", shuffle=False)
    b = next(iter(ds))
    assert b["x_avail"][:, 1].sum() == 0 and b["y_avail"][:, 1].sum() == 0
    assert torch.all(b["x_data"][:, :, 1] == -1.0)
    assert b["x_avail"][:, 0].sum() > 0


# ---------------------------------------------------------------------------
# (3) the meta_dsf{d}[0] == meta_dsf1[0] - log2(d) invariant
# ---------------------------------------------------------------------------

def test_verify_accepts_a_clean_h5(v2_h5):
    from candi_kit.prep.bake import _verify
    _verify(v2_h5, allow_missing_control=False)


def test_verify_catches_dsf1_masquerading_as_dsf8(tmp_path):
    """E3's failure mode seen from the other end: an assay absent from an explicit DSF-8 map used to
    load DSF-1 counts and pair them with DSF-1 depth metadata, so counts_dsf8 held a lie that the
    depth-steering objective then learned from."""
    from candi_kit.prep.bake import _verify

    p = write_v2_h5(tmp_path / "masq.h5")
    with h5py.File(p, "r+") as h5:
        md1 = np.array(h5["biosamples"]["T_a"]["meta_dsf1"])
        md8 = h5["biosamples"]["T_a"]["meta_dsf8"]
        md8[0, 1] = md1[0, 1]                       # column 1 claims DSF-1 depth at DSF-8
    with pytest.raises((AssertionError, ValueError)):
        _verify(p, allow_missing_control=False)


# ---------------------------------------------------------------------------
# (4) int16 overflow  /  (5) unparseable run_type
# ---------------------------------------------------------------------------

def test_count_past_int16_is_stored_not_clipped(tmp_path):
    """E4: counts are stored int32 and survive past the int16 ceiling.

    The research pipeline stored int16 and CLIPPED silently above 32,767. Real EIC data exceeds it --
    B_DND-41/DNase-seq reaches 52,051 in a cCRE-sampled window -- so the ceiling corrupted real training
    data. This asserts the value round-trips intact rather than that the bake refuses it.
    """
    pytest.importorskip("pysam", reason="the bake needs the [prepare] extra")
    import h5py
    from candi_kit.prep.bake import bake

    root = make_root(tmp_path / "root", huge_count_at=("T_a", "assay1"))
    side = make_side(tmp_path)
    out = tmp_path / "out" / "ovf.h5"
    bake(root, make_panel(), out, side, max_tile_per_chrom=2)

    with h5py.File(out, "r") as h:
        ds = h["biosamples"]["T_a"]["counts_dsf1"]
        assert ds.dtype == np.int32, f"counts must be int32, got {ds.dtype}"
        assert int(np.asarray(ds).max()) > 32767, "the >int16 value was clipped or never written"


def test_unparseable_run_type_raises(tmp_path):
    """E5: the if/elif had no else and `runt` persisted across the assay loop, so an unparseable value
    silently inherited the PREVIOUS assay's run_type."""
    pytest.importorskip("pysam", reason="the bake needs the [prepare] extra")
    from candi_kit.prep.bake import bake

    root = make_root(tmp_path / "root", run_types={("T_a", "assay1"): "interleaved"})
    side = make_side(tmp_path)
    with pytest.raises(ValueError) as ei:
        bake(root, make_panel(), tmp_path / "out" / "rt.h5", side, max_tile_per_chrom=2)
    assert "run_type" in str(ei.value)


# ---------------------------------------------------------------------------
# (6) h5 -> dataset -> model round-trip at a NON-q19 scale
# ---------------------------------------------------------------------------

def test_h5_roundtrips_to_model_at_3_assays_384_bins(v2_h5):
    from candi_kit.batch import make_masker, prepare_masked_batch
    from candi_kit.dataset import CandiKitH5Dataset
    from candi_kit.model import build_real_model, forward_full

    ds = CandiKitH5Dataset(v2_h5, "type1", train=True, batch_size=2, dsf_sampling="uniform",
                           shuffle=False)
    assert ds.num_assays == 3 and ds.context_bins == CTX_BINS and ds.resolution == RES
    assert tuple(ds.dsf_list) == DSFS

    torch.manual_seed(0)
    model = build_real_model(embed_dim=16, n_transformer_layers=1, feat_per_assay=8,
                             num_assays=ds.num_assays, context_length=ds.context_bins).eval()
    prepared = prepare_masked_batch(next(iter(ds)), make_masker(p_full_assay=1.0), torch.device("cpu"))
    assert prepared is not None
    assert prepared["x_data"].shape[2] == ds.num_assays + 1        # control appended at index A
    with torch.no_grad():
        out = forward_full(model, prepared)
    assert out["mu"].shape == (prepared["x_data"].shape[0], CTX_BINS, ds.num_assays)
    assert torch.isfinite(out["mu"]).all()


# ---------------------------------------------------------------------------
# (7) the loader hands the NB head raw integer counts
# ---------------------------------------------------------------------------

def test_loader_yields_raw_nonnegative_integer_counts(v2_h5):
    """arcsinh is applied INSIDE V2Encoder (signal_transform='arcsinh') and never by the counts loader.
    A double-arcsinh collapses the range and the NB targets stop being counts."""
    from candi_kit.dataset import CandiKitH5Dataset

    ds = CandiKitH5Dataset(v2_h5, "type1", train=True, batch_size=2, dsf_sampling="off", shuffle=False)
    b = next(iter(ds))
    y, avail = b["y_data"], b["y_avail"] > 0
    vals = y[avail.unsqueeze(1).expand_as(y)]
    assert vals.numel() > 0
    assert torch.all(vals >= 0)
    assert torch.all(vals == vals.round())
    assert float(vals.max()) > 1.0        # a double-arcsinh would sit near log-scale magnitudes
