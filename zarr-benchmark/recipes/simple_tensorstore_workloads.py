"""
`pip install tensorstore`
"""

from pathlib import Path

import numpy as np
import simple_datasets
import tensorstore as ts
from perfcapture.dataset import Dataset
from perfcapture.metrics import MetricsForRun
from perfcapture.workload import Workload


class TensorStoreLoadEntireArray(Workload):
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
        z = _tensorstore_load_zarr(dataset_path)
        return MetricsForRun(
            nbytes_in_final_array=z.nbytes,
        )


def _tensorstore_load_zarr(dataset_path: Path) -> np.ndarray:
    """Loads a Zarr v3 dataset using TensorStore."""
    spec = {
        "driver": "zarr3",  # Zarr v3 requires explicit version in TensorStore
        "kvstore": {
            "driver": "file",
            "path": str(dataset_path),
        },
        "metadata_key": ".zmetadata",  # Required for Zarr v3 metadata structure
        "metadata_encoding": "json",  # Ensures compatibility with Zarr v3 JSON metadata
    }
    
    ds = ts.open(spec).result()  # Open the dataset
    ds = ds.read().result()  # Read the dataset into memory
    return ds
