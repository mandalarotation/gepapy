import cupy as cp  # type: ignore
import numpy as np  # type: ignore
from gen_scheduling_cuda.operations import Operations as Operations
from typing import Optional, Union

class Flow_Shop(Operations):
    def __init__(
        self,
        processing_time: Optional[Union[list, np.ndarray, cp.core.core.ndarray]],
        machine_sequence: Optional[Union[list, np.ndarray, cp.core.core.ndarray]],
        due_date: Optional[Union[list, np.ndarray, cp.core.core.ndarray]],
        weights: Optional[Union[list, np.ndarray, cp.core.core.ndarray]],
        n_samples: int,
        n_jobs: int,
        n_operations: int,
        n_machines: int,
        percent_cross: float = ...,
        percent_intra_cross: float = ...,
        percent_mutation: float = ...,
        percent_intra_mutation: float = ...,
        percent_migration: float = ...,
        percent_selection: float = ...,
        fitness_type: str = ...,
    ) -> None: ...
