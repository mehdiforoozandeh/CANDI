import json
import os
from pathlib import Path

import numpy as np
import torch

from data import CANDIIterableDataset

try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency at import time
    h5py = None


H5_METADATA_DIRNAME = "h5_metadata"
H5_MANIFEST_FILENAME = "manifest.json"
H5_BIOSAMPLE_STORES_DIRNAME = "biosamples_h5"
H5_SCHEMA_VERSION = 1


class _H5ArrayView:
    """
    Sliceable view over a 1D HDF5 dataset or one column of a 2D dataset.
    """

    def __init__(self, dataset, column_idx=None, transform=None, dtype=None):
        self.dataset = dataset
        self.column_idx = column_idx
        self.transform = transform
        self.dtype = dtype

    def _apply(self, values):
        values = np.asarray(values)
        if self.transform is not None:
            values = self.transform(values)
        if self.dtype is not None:
            values = values.astype(self.dtype, copy=False)
        return values

    def __getitem__(self, item):
        if self.column_idx is None:
            values = self.dataset[item]
        else:
            values = self.dataset[item, self.column_idx]
        return self._apply(values)


class H5CANDIDataHandler:
    """
    HDF5-backed variant of the prepared-data handler.

    It mirrors the prepared-data contract from `data_zarr.py`, but stores each
    biosample in a single `.h5` file to avoid inode explosions on `/project`.
    """

    def __init__(self, *args, **kwargs):
        self.prepared_metadata_path = kwargs.pop("prepared_metadata_path", None)
        self.data_backend = kwargs.pop("data_backend", "h5")
        super().__init__(*args, **kwargs)
        self._init_h5_store()

    def _init_h5_store(self):
        if h5py is None:
            raise ImportError(
                "HDF5 backend requested, but `h5py` is not installed. "
                "Install it in the active environment before using `data_h5.py`."
            )

        if self.data_backend != "h5":
            raise ValueError(f"Unsupported data backend for H5 handler: {self.data_backend}")

        self.prepared_metadata_path = self.prepared_metadata_path or os.path.join(self.base_path, H5_METADATA_DIRNAME)
        self.prepared_biosample_stores_path = os.path.join(self.base_path, H5_BIOSAMPLE_STORES_DIRNAME)
        self.manifest_path = os.path.join(self.base_path, H5_MANIFEST_FILENAME)

        if not os.path.isdir(self.prepared_metadata_path):
            raise FileNotFoundError(f"Prepared HDF5 metadata directory not found: {self.prepared_metadata_path}")
        if not os.path.isdir(self.prepared_biosample_stores_path):
            raise FileNotFoundError(
                f"Prepared biosample HDF5 directory not found: {self.prepared_biosample_stores_path}"
            )
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Prepared manifest not found: {self.manifest_path}")

        with open(self.manifest_path, "r") as handle:
            self.store_manifest = json.load(handle)

        self._bios_handles = {}
        self._bios_metadata_cache = {}
        self._array_cache = {}

        manifest_dataset_type = self.store_manifest.get("dataset_type")
        if manifest_dataset_type and manifest_dataset_type != self.dataset_type:
            raise ValueError(
                f"Prepared store dataset_type={manifest_dataset_type} does not match requested {self.dataset_type}"
            )
        manifest_resolution = self.store_manifest.get("resolution")
        if manifest_resolution and int(manifest_resolution) != int(self.resolution):
            raise ValueError(
                f"Prepared store resolution={manifest_resolution} does not match requested {self.resolution}"
            )

    def _bios_store_path(self, bios_name):
        return os.path.join(self.prepared_biosample_stores_path, f"{bios_name}.h5")

    def _get_bios_handle(self, bios_name):
        if bios_name not in self._bios_handles:
            bios_store_path = self._bios_store_path(bios_name)
            if not os.path.exists(bios_store_path):
                raise FileNotFoundError(f"Prepared biosample store missing for {bios_name}: {bios_store_path}")
            self._bios_handles[bios_name] = h5py.File(bios_store_path, "r")
        return self._bios_handles[bios_name]

    def _metadata_file(self, bios_name):
        return os.path.join(self.prepared_metadata_path, f"{bios_name}.json")

    def _load_bios_prepared_metadata(self, bios_name):
        if bios_name not in self._bios_metadata_cache:
            metadata_path = self._metadata_file(bios_name)
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Prepared metadata missing for biosample {bios_name}: {metadata_path}")
            with open(metadata_path, "r") as handle:
                self._bios_metadata_cache[bios_name] = json.load(handle)
        return self._bios_metadata_cache[bios_name]

    def _get_array(self, bios_name, array_path):
        cache_key = (bios_name, array_path)
        if cache_key not in self._array_cache:
            self._array_cache[cache_key] = self._get_bios_handle(bios_name)[array_path]
        return self._array_cache[cache_key]

    def _counts_array_path(self, dsf, chrom):
        return f"counts/dsf_{dsf}/{chrom}"

    def _signal_array_path(self, modality, chrom):
        return f"{modality}/{chrom}"

    def _control_array_path(self, dsf, chrom):
        return f"control/dsf_{dsf}/{chrom}"

    @staticmethod
    def _safe_log1p(values):
        return np.log1p(values)

    @staticmethod
    def _safe_arcsinh(values):
        return np.arcsinh(values)

    def _signal_transform_fn(self):
        if self.signal_transform == "arcsinh":
            return self._safe_arcsinh
        if self.signal_transform == "log1p":
            return self._safe_log1p
        return None

    def _slice_from_view(self, array_view, locus):
        if len(locus) == 1:
            return array_view
        start_bin = int(locus[1]) // self.resolution
        end_bin = int(locus[2]) // self.resolution
        return array_view[start_bin:end_bin]

    def _get_available_assay_dsfs(self, bios_name, assay, chrom, f_format="npz"):
        del f_format
        bios_meta = self._load_bios_prepared_metadata(bios_name)
        available = []
        for dsf in self.dsf_list:
            dsf_meta = bios_meta.get("counts", {}).get(str(dsf))
            if dsf_meta is None:
                continue
            chrom_map = dsf_meta.get("chrom_assay_to_idx", {}).get(chrom, {})
            if assay not in chrom_map:
                continue
            if chrom not in dsf_meta.get("chromosomes", []):
                continue
            available.append(dsf)
        return available

    def _get_available_control_dsfs(self, bios_name, chrom, f_format="npz"):
        del f_format
        bios_meta = self._load_bios_prepared_metadata(bios_name)
        available = []
        for dsf in self.dsf_list:
            dsf_meta = bios_meta.get("control", {}).get(str(dsf))
            if dsf_meta is None:
                continue
            if chrom not in dsf_meta.get("chromosomes", []):
                continue
            available.append(dsf)
        return available

    def load_bios_Counts(self, bios_name, locus, DSF=1, f_format="npz"):
        del f_format
        exps = self._get_count_experiments(bios_name)
        chrom = str(locus[0])
        bios_meta = self._load_bios_prepared_metadata(bios_name)
        loaded_data = {}
        loaded_metadata = {}

        for assay in exps:
            assay_dsf = DSF.get(assay, 1) if isinstance(DSF, dict) else DSF
            dsf_meta = bios_meta.get("counts", {}).get(str(assay_dsf))
            if dsf_meta is None:
                continue

            assay_to_idx = dsf_meta.get("chrom_assay_to_idx", {}).get(chrom, {})
            if assay not in assay_to_idx or chrom not in dsf_meta.get("chromosomes", []):
                continue

            array = self._get_array(bios_name, self._counts_array_path(assay_dsf, chrom))
            assay_idx = int(assay_to_idx[assay])
            view = _H5ArrayView(array, column_idx=assay_idx, dtype=np.int16)
            loaded_data[assay] = self._slice_from_view(view, locus)
            loaded_metadata[assay] = dsf_meta["metadata"][assay]

        return loaded_data, loaded_metadata

    def load_bios_BW(self, bios_name, locus, f_format="npz", arcsinh=True):
        del f_format
        exps = list(self.navigation[bios_name].keys())
        if "RNA-seq" in exps:
            exps.remove("RNA-seq")
        if "chipseq-control" in exps:
            exps.remove("chipseq-control")

        chrom = str(locus[0])
        bios_meta = self._load_bios_prepared_metadata(bios_name)
        assay_to_idx = bios_meta.get("pval", {}).get("chrom_assay_to_idx", {}).get(chrom, {})
        available_chroms = bios_meta.get("pval", {}).get("chromosomes", [])
        loaded_data = {}

        if chrom not in available_chroms:
            return loaded_data

        transform = self._signal_transform_fn() if arcsinh else None
        array = self._get_array(bios_name, self._signal_array_path("pval", chrom))
        for assay in exps:
            if assay not in assay_to_idx:
                continue
            view = _H5ArrayView(array, column_idx=int(assay_to_idx[assay]), transform=transform, dtype=np.float16)
            loaded_data[assay] = self._slice_from_view(view, locus)

        return loaded_data

    def load_bios_Peaks(self, bios_name, locus, f_format="npz"):
        del f_format
        exps = list(self.navigation[bios_name].keys())
        if "RNA-seq" in exps:
            exps.remove("RNA-seq")
        if "chipseq-control" in exps:
            exps.remove("chipseq-control")

        chrom = str(locus[0])
        bios_meta = self._load_bios_prepared_metadata(bios_name)
        assay_to_idx = bios_meta.get("peaks", {}).get("chrom_assay_to_idx", {}).get(chrom, {})
        available_chroms = bios_meta.get("peaks", {}).get("chromosomes", [])
        loaded_data = {}

        if chrom not in available_chroms:
            return loaded_data

        array = self._get_array(bios_name, self._signal_array_path("peaks", chrom))
        for assay in exps:
            if assay not in assay_to_idx:
                continue
            view = _H5ArrayView(array, column_idx=int(assay_to_idx[assay]), dtype=np.float16)
            loaded_data[assay] = self._slice_from_view(view, locus)

        return loaded_data

    def load_bios_Control(self, bios_name, locus, DSF=1, f_format="npz"):
        del f_format
        chrom = str(locus[0])
        bios_meta = self._load_bios_prepared_metadata(bios_name)
        dsf_meta = bios_meta.get("control", {}).get(str(DSF))
        if dsf_meta is None or chrom not in dsf_meta.get("chromosomes", []):
            return {}, {}

        array = self._get_array(bios_name, self._control_array_path(DSF, chrom))
        view = _H5ArrayView(array, dtype=np.int16)
        loaded_data = {"chipseq-control": self._slice_from_view(view, locus)}
        loaded_metadata = {"chipseq-control": dsf_meta["metadata"]}
        return loaded_data, loaded_metadata

    def close(self):
        for handle in self._bios_handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._bios_handles = {}

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class H5CANDIIterableDataset(H5CANDIDataHandler, CANDIIterableDataset):
    """
    HDF5-backed iterable dataset. Keeps the existing iterator and batch contract from
    `CANDIIterableDataset`, but with storage access redirected to per-biosample HDF5 files.
    """

    def __init__(self, **kwargs):
        CANDIIterableDataset.__init__(
            self,
            base_path=kwargs.get("base_path"),
            resolution=kwargs.get("resolution", 25),
            dataset_type=kwargs.get("dataset_type", "merged"),
            DNA=kwargs.get("DNA", True),
            bios_batchsize=1,
            loci_batchsize=1,
            dsf_list=kwargs.get("dsf_list", [1, 2]),
            signal_transform=kwargs.get("signal_transform", "arcsinh"),
            enable_per_assay_dsf_sampling=kwargs.get("enable_per_assay_dsf_sampling", False),
            per_assay_dsf_sampling_mode=kwargs.get("per_assay_dsf_sampling_mode", "uniform"),
        )
        self.prepared_metadata_path = kwargs.get("prepared_metadata_path")
        self.data_backend = kwargs.get("data_backend", "h5")
        self._init_h5_store()
        self.kwargs = kwargs
        self.fill_prompt_mode = kwargs.get("fill_prompt_mode", "sample")
        print(
            f"H5CANDIIterableDataset initialized with bios_batchsize={self.bios_batchsize}, "
            f"loci_batchsize={self.loci_batchsize}, dsf_list={self.dsf_list}"
        )
        print(f"Signal transformation: {self.signal_transform}")
        print(f"Fill-in-prompt mode: {self.fill_prompt_mode}")


def get_prepared_eic_h5_path(base_path):
    """
    Resolve the default sister-directory prepared root for the EIC dataset.
    """
    source_root = Path(base_path).expanduser().resolve()
    if source_root.name == "DATA_CANDI_EIC":
        return str(source_root.with_name("DATA_CANDI_EIC_H5"))
    return str(source_root.parent / f"{source_root.name.rstrip('/')}_H5")
