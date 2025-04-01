"""Simple Zarr Python workloads (Zarr v3 compatible)."""

from pathlib import Path
import simple_datasets
import zarr
from perfcapture.dataset import Dataset
from perfcapture.metrics import MetricsForRun
from perfcapture.workload import Workload


class ZarrPythonLoadEntireArray(Workload):
    def init_datasets(self) -> tuple[Dataset, ...]:
        return (
            simple_datasets.Uncompressed_1_Chunk(),
            simple_datasets.LZ4_200_Chunks(),
            simple_datasets.Uncompressed_200_Chunks(),
            simple_datasets.LZ4_20000_Chunks(),
            simple_datasets.Uncompressed_20000_Chunks(),
        )

    def run(self, dataset_path: Path) -> MetricsForRun:
        """Load entire Zarr v3 dataset into RAM."""
        z = _zarr_v3_load_array(dataset_path)
        return MetricsForRun(
            nbytes_in_final_array=z.nbytes,
        )


def _zarr_v3_load_array(dataset_path: Path) -> zarr.Array:
    """Load a Zarr v3 array from the given path."""
    store = zarr.storage.DirectoryStore(str(dataset_path))
    group = zarr.open_group(store, mode='r')  # Open as a Zarr v3 group
    keys = list(group.array_keys())  # Get all available arrays

    if not keys:
        raise ValueError(f"No arrays found in Zarr v3 dataset at {dataset_path}")

    array = group[keys[0]]  # Load the first available array
    return array[:]
