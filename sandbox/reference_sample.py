"""
Build training tensors exactly like CANDIDataHandler.get_batch / IterableDataset.

Used by prepare_h5 (bake + parity) and tests. All logic delegates to data.py methods.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from data import CANDIDataHandler

from sandbox import SANDBOX_ASSAYS


def make_sandbox_handler(
    eic_data_root: str,
    *,
    includes: List[str] | None = None,
    dsf_list: Tuple[int, ...] = (1, 2, 4, 8),
    genomic_coords_mode: str = "val",
) -> CANDIDataHandler:
    """EIC handler with navigation+aliases restricted to sandbox panel assays."""
    includes = includes or list(SANDBOX_ASSAYS)
    h = CANDIDataHandler(
        base_path=eic_data_root,
        resolution=25,
        dataset_type="eic",
        DNA=True,
        dsf_list=list(dsf_list),
        signal_transform="arcsinh",
    )
    h._filter_navigation(includes, [])
    h._build_assay_to_id_mapping()
    # Train mode omits chr21 from chr_sizes; sandbox eval+bake needs chr21.
    h._load_genomic_coords(mode=genomic_coords_mode)
    return h


def restrict_navigation_to_biosamples(handler: CANDIDataHandler, bios_full_names: List[str]) -> None:
    """Drop all navigation keys except the listed full biosample names (e.g. T_K562)."""
    allow = set(bios_full_names)
    for k in list(handler.navigation.keys()):
        if k not in allow:
            del handler.navigation[k]


def _assay_order(handler: CANDIDataHandler) -> List[str]:
    return list(handler.aliases["experiment_aliases"].keys())


def dsf_maps_to_tensors(handler: CANDIDataHandler, x_map: Dict[str, int], y_map: Dict[str, int]):
    """Maps assay_name -> dsf int to [F] tensors aligned with experiment_aliases (-1 if missing)."""
    xs = []
    ys = []
    for a in _assay_order(handler):
        xs.append(int(x_map.get(a, -1)))
        ys.append(int(y_map.get(a, -1)))
    return torch.tensor(xs, dtype=torch.int64), torch.tensor(ys, dtype=torch.int64)


def reference_tensors(
    handler: CANDIDataHandler,
    bios_name: str,
    locus: List,
    x_dsf_map: Dict[str, int],
    y_dsf_map: Dict[str, int],
    *,
    y_prompt: bool = True,
    random_shift: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    locus: [chrom, start, end] in bp, aligned to resolution grid like prod.

    Returns keys compatible with sandbox training:
      y_data, y_meta, y_avail, y_pval, y_peaks, x_data, x_meta, x_avail,
      x_dna, control_data, control_meta, control_avail, x_dsf, y_dsf
    """
    assert len(locus) == 3, "expected [chrom, start, end]"
    chrom = locus[0]
    # make_bios_tensor_Control uses self.context_length when control is missing
    start, end = int(locus[1]), int(locus[2])
    handler.context_length = end - start
    # Prod loads whole chromosome then slices per locus (see CANDIDataHandler.new_epoch).
    chrom_only = [chrom]

    xd, xmd = handler.load_bios_Counts(bios_name, chrom_only, x_dsf_map)
    yd, ymd = handler.load_bios_Counts(bios_name, chrom_only, y_dsf_map)

    xreg = handler._select_region_from_loaded_data(xd, locus)
    yreg = handler._select_region_from_loaded_data(yd, locus)

    x_data, x_meta, x_avail = handler.make_region_tensor_Counts([xreg], [xmd])
    y_data, y_meta, y_avail = handler.make_region_tensor_Counts([yreg], [ymd])

    # fill_in_prompt default (fill_missing=False) returns md unchanged — keep prod behavior.
    if y_prompt:
        y_meta = handler.fill_in_prompt(y_meta, sample=True)

    pfull = handler.load_bios_BW(bios_name, chrom_only, arcsinh=True)
    preg = handler._select_region_from_loaded_data(pfull, locus)
    y_pval, _ = handler.make_region_tensor_BW([preg])

    peaksfull = handler.load_bios_Peaks(bios_name, chrom_only)
    pregreg = handler._select_region_from_loaded_data(peaksfull, locus)
    y_peaks, _ = handler.make_region_tensor_Peaks([pregreg])

    one_hot = handler._get_cached_onehot_for_locus(locus)
    if random_shift:
        # training path may shift; parity uses False
        one_hot = handler._onehot_for_locus(locus)
    x_dna = one_hot.unsqueeze(0)

    cd, cmd = handler.load_bios_Control(bios_name, chrom_only, DSF=1)
    creg = handler._select_region_from_loaded_data(cd, locus)
    control_data, control_meta, control_avail = handler.make_region_tensor_Control([creg], [cmd])

    x_dsf, y_dsf = dsf_maps_to_tensors(handler, x_dsf_map, y_dsf_map)

    return {
        "x_data": x_data,
        "x_meta": x_meta,
        "x_avail": x_avail,
        "x_dna": x_dna,
        "control_data": control_data,
        "control_meta": control_meta,
        "control_avail": control_avail,
        "y_data": y_data,
        "y_meta": y_meta,
        "y_avail": y_avail,
        "y_pval": y_pval,
        "y_peaks": y_peaks,
        "x_dsf": x_dsf.unsqueeze(0),
        "y_dsf": y_dsf.unsqueeze(0),
    }


def dsf_tensors(handler: CANDIDataHandler, x_map: Dict[str, int], y_map: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    return dsf_maps_to_tensors(handler, x_map, y_map)


def fixed_dsf_pair_maps(
    handler: CANDIDataHandler,
    bios_name: str,
    chrom: str,
    x_dsf: int,
    y_dsf: int,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Per-assay maps: use x_dsf / y_dsf only where that DSF exists on disk; else skip assay (omitted from maps -> load_bios skips)."""
    x_map: Dict[str, int] = {}
    y_map: Dict[str, int] = {}
    for assay in _assay_order(handler):
        if assay not in handler.navigation.get(bios_name, {}):
            continue
        av = handler._get_available_assay_dsfs(bios_name, assay, chrom)
        if not av:
            continue
        if x_dsf in av:
            x_map[assay] = x_dsf
        if y_dsf in av:
            y_map[assay] = y_dsf
    return x_map, y_map
