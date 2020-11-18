import cupy as cp  # type: ignore
import numpy as np  # type: ignore
from typing import Optional as Optional
from typing import Union as Union
from gen_scheduling_cuda.kernels.crossA0001 import CrossA0001 as CrossA0001
from gen_scheduling_cuda.kernels.fitnessA0001 import FitnessA0001 as FitnessA0001
from gen_scheduling_cuda.kernels.mutationA0001 import MutationA0001 as MutationA0001
from gen_scheduling_cuda.kernels.permutationA0001 import (
    PermutationA0001 as PermutationA0001,
)
from gen_scheduling_cuda.kernels.sortA0001 import SortA0001 as SortA0001

class Operations(PermutationA0001, CrossA0001, MutationA0001, SortA0001, FitnessA0001):
    def __init__(self) -> None: ...
    def set_n_samples(self, n_samples: int) -> int: ...
    def _set_n_jobs(self, n_jobs: int) -> Optional[Union[None, int]]: ...
    def _set_n_machines(self, n_machines: int) -> Optional[Union[None, int]]: ...
    def _set_n_operations(self, n_operations: int) -> Optional[Union[None, int]]: ...
    def set_fitness_type(self, fitness_type: str) -> Optional[Union[None, str]]: ...
    def _set_percent_cross(self, percent_cross: float) -> float: ...
    def set_percent_intra_cross(self, percent_intra_cros: float) -> float: ...
    def _set_percent_mutation(self, percent_mutation: float) -> float: ...
    def set_percent_intra_mutation(self, percent_intra_mutation: float) -> float: ...
    def _set_percent_migration(self, percent_selection: float) -> float: ...
    def _set_percent_selection(self, percent_migration: float) -> float: ...
    def _set_fitness(
        self, fitness: cp.core.core.ndarray
    ) -> Optional[Union[None, cp.core.core.ndarray]]: ...
    def set_population(
        self, population: Optional[Union[cp.core.core.ndarray, None]] = None
    ) -> Optional[Union[None, cp.core.core.ndarray]]: ...
    def _set_processing_time(
        self, processing_time: Optional[Union[list, np.ndarray, cp.core.core.ndarray]]
    ) -> cp.core.core.ndarray: ...
    def _set_machine_sequence(
        self, machine_sequence: Optional[Union[list, np.ndarray, cp.core.core.ndarray]]
    ) -> cp.core.core.ndarray: ...
    def _set_due_date(
        self, due_date: Optional[Union[list, np.ndarray, cp.core.core.ndarray]]
    ) -> cp.core.core.ndarray: ...
    def _set_weights(
        self, weights: Optional[Union[list, np.ndarray, cp.core.core.ndarray]]
    ) -> cp.core.core.ndarray: ...
    def get_n_samples(self) -> int: ...
    def get_n_jobs(self) -> int: ...
    def get_n_machines(self) -> int: ...
    def get_n_operations(self) -> int: ...
    def get_fitness_type(self) -> cp.core.core.ndarray: ...
    def get_percent_cross(self) -> float: ...
    def get_percent_intra_cross(self) -> float: ...
    def get_percent_mutation(self) -> float: ...
    def get_percent_intra_mutation(self) -> float: ...
    def get_percent_migration(self) -> float: ...
    def get_percent_selection(self) -> float: ...
    def get_fitness(self) -> cp.core.core.ndarray: ...
    def get_population(self) -> cp.core.core.ndarray: ...
    def get_processing_time(self) -> cp.core.core.ndarray: ...
    def get_machine_sequence(self) -> cp.core.core.ndarray: ...
    def get_due_date(self) -> cp.core.core.ndarray: ...
    def get_weights(self) -> cp.core.core.ndarray: ...
    def exec_permutationA0001(self) -> cp.core.core.ndarray: ...
    def exec_crossA0001(self) -> None: ...
    def exec_mutationA0001(self) -> None: ...
    def exec_migrationA0001(self) -> None: ...
    def exec_sortA0001(self) -> None: ...
    def exec_fitnessA0001(self) -> None: ...
    def get_plan(self, row: int, fact_conv: int, start_time: int) -> list: ...
