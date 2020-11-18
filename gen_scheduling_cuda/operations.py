from __future__ import division
from typing import Optional, Union
import numpy as np  # type: ignore
import cupy as cp  # type: ignore


from gen_scheduling_cuda.exceptions.operations import Check
from gen_scheduling_cuda.kernels.permutationA0001 import PermutationA0001
from gen_scheduling_cuda.kernels.crossA0001 import CrossA0001
from gen_scheduling_cuda.kernels.mutationA0001 import MutationA0001
from gen_scheduling_cuda.kernels.sortA0001 import SortA0001
from gen_scheduling_cuda.kernels.fitnessA0001 import FitnessA0001


class Operations(PermutationA0001, CrossA0001, MutationA0001, SortA0001, FitnessA0001):
    """Operations."""

    def __init__(self) -> None:
        """__init__.

        :rtype: None
        """
        self._initialized: bool
        self._n_samples: int
        self._n_jobs: int
        self._n_machines: int
        self._n_operations: int
        self._fitness_type: cp.core.core.ndarray
        self._percent_cross: float
        self._percent_intra_cross: float
        self._percent_mutation: float
        self._percent_intra_mutation: float
        self._percent_migration: float
        self._percent_selection: float
        self._fitness: cp.core.core.ndarray
        self._population: cp.core.core.ndarray
        self._processing_time: cp.core.core.ndarray
        self._machine_sequence: cp.core.core.ndarray
        self._due_date: cp.core.core.ndarray
        self._weights: cp.core.core.ndarray

    @Check._set_n_samples_check
    def set_n_samples(self, n_samples: int) -> Optional[Union[None, int]]:
        """set_n_samples.

        :param n_samples:
        :type n_samples: int
        :rtype: int
        """
        if self._initialized:
            self._n_samples = n_samples
            return None
        else:
            return n_samples

    @Check._set_n_jobs_check
    def _set_n_jobs(self, n_jobs: int) -> Optional[Union[None, int]]:
        """_set_n_jobs.

        :param n_jobs:
        :type n_jobs: int
        :rtype: int
        """
        if self._initialized:
            self._n_jobs = n_jobs
            return None
        else:
            return n_jobs

    @Check._set_n_machines_check
    def _set_n_machines(self, n_machines: int) -> Optional[Union[None, int]]:
        """_set_n_machines.

        :param n_machines:
        :type n_machines: int
        :rtype: int
        """
        if self._initialized:
            self._n_machines = n_machines
            return None
        else:
            return n_machines

    @Check._set_n_operations_check
    def _set_n_operations(self, n_operations: int) -> Optional[Union[None, int]]:
        """_set_n_operations.

        :param n_operations:
        :type n_operations: int
        :rtype: int
        """
        if self._initialized:
            self._n_operations = n_operations
            return None
        else:
            return n_operations

    @Check._set_fitness_type_check
    def set_fitness_type(self, fitness_type: str) -> Optional[Union[None, str]]:
        """_set_fitness_type.

        :param fitness_type:
        :type fitness_type: str
        :rtype: str
        """
        if self._initialized:
            self._fitness_type = fitness_type
            return None
        else:
            return fitness_type

    @Check._set_percent_cross_check
    def _set_percent_cross(self, percent_cross: float) -> Optional[Union[None, float]]:
        """set_percent_cross.

        :param percent_cross:
        :type percent_cross: float
        :rtype: float
        """
        if self._initialized:
            self._percent_cross = percent_cross
            return None
        else:
            return percent_cross

    @Check._set_percent_intra_cross_check
    def set_percent_intra_cross(
        self, percent_intra_cross: float
    ) -> Optional[Union[None, float]]:
        """set_percent_intra_cross.

        :param percent_intra_cros:
        :type percent_intra_cros: float
        :rtype: float
        """
        if self._initialized:
            self._percent_intra_cross = percent_intra_cross
            return None
        else:
            return percent_intra_cross

    @Check._set_percent_mutation_check
    def _set_percent_mutation(
        self, percent_mutation: float
    ) -> Optional[Union[None, float]]:
        """set_percent_mutation.

        :param percent_mutation:
        :type percent_mutation: float
        :rtype: float
        """
        if self._initialized:
            self._percent_mutation = percent_mutation
            return None
        else:
            return percent_mutation

    @Check._set_percent_intra_mutation_check
    def set_percent_intra_mutation(
        self, percent_intra_mutation: float
    ) -> Optional[Union[None, float]]:
        """set_percent_intra_mutation.

        :param percent_intra_mutation:
        :type percent_intra_mutation: float
        :rtype: float
        """
        if self._initialized:
            self._percent_intra_mutation = percent_intra_mutation
            return None
        else:
            return percent_intra_mutation

    @Check._set_percent_migration_check
    def _set_percent_migration(
        self, percent_selection: float
    ) -> Optional[Union[None, float]]:
        """set_percent_migration.

        :param percent_selection:
        :type percent_selection: float
        :rtype: float
        """
        if self._initialized:
            self._percent_selection = percent_selection
            return None
        else:
            return percent_selection

    @Check._set_percent_selection_check
    def _set_percent_selection(
        self, percent_migration: float
    ) -> Optional[Union[None, float]]:
        """set_percent_selection.

        :param percent_migration:
        :type percent_migration: float
        :rtype: float
        """
        if self._initialized:
            self._percent_migration = percent_migration
            return None
        else:
            return percent_migration

    @Check._set_fitness_check
    def _set_fitness(
        self, fitness: cp.core.core.ndarray
    ) -> Optional[Union[None, cp.core.core.ndarray]]:
        """_set_fitness.

        :param fitness:
        :type fitness: cp.core.core.ndarray
        :rtype: None
        """
        if self._initialized:
            self._fitness = fitness
            return None
        else:
            return fitness

    @Check._set_population_check
    def set_population(
        self, population: Optional[Union[cp.core.core.ndarray, None]] = None
    ) -> Optional[Union[None, cp.core.core.ndarray]]:
        """_set_population.

        :param population:
        :type population: Optional[Union[cp.core.core.ndarray, None]]
        """
        if self._initialized:
            self._population = population
            return None
        else:
            return self.exec_permutationA0001()

    @Check._set_processing_time_check
    def _set_processing_time(
        self, processing_time: Optional[Union[list, np.ndarray, cp.core.core.ndarray]]
    ) -> cp.core.core.ndarray:
        """_set_processing_time.

        :param processing_time:
        :type processing_time: Optional[Union[list, np.ndarray, cp.core.core.ndarray]]
        :rtype: cp.core.core.ndarray
        """
        return cp.array(processing_time, dtype=cp.float32)

    @Check._set_machine_sequence_check
    def _set_machine_sequence(
        self, machine_sequence: Optional[Union[list, np.ndarray, cp.core.core.ndarray]]
    ) -> cp.core.core.ndarray:
        """_set_machine_sequence.

        :param machine_sequence:
        :type machine_sequence: Optional[Union[list, np.ndarray, cp.core.core.ndarray]]
        :rtype: cp.core.core.ndarray
        """
        return cp.array(machine_sequence, dtype=cp.float32)

    @Check._set_due_date_check
    def _set_due_date(
        self, due_date: Optional[Union[list, np.ndarray, cp.core.core.ndarray]]
    ) -> cp.core.core.ndarray:
        """_set_due_date.

        :param due_date:
        :type due_date: Optional[Union[list, np.ndarray, cp.core.core.ndarray]]
        :rtype: cp.core.core.ndarray
        """
        due_date = cp.array(due_date, dtype=cp.float32)
        return due_date

    @Check._set_weights_check
    def _set_weights(
        self, weights: Optional[Union[list, np.ndarray, cp.core.core.ndarray]]
    ) -> cp.core.core.ndarray:
        """_set_weights.

        :param weights:
        :type weights: Optional[Union[list, np.ndarray, cp.core.core.ndarray]]
        :rtype: cp.core.core.ndarray
        """
        weights = cp.array(weights, dtype=cp.float32)
        return weights

    @Check._set_percents_c_m_m_s_check
    def set_percents_c_m_m_s(
        self,
        percent_cross: float,
        percent_mutation: float,
        percent_migration: float,
        percent_selection: float,
    ) -> None:
        """set_percents_c_m_m_s.

        :param percent_cross:
        :type percent_cross: float
        :param percent_mutation:
        :type percent_mutation: float
        :param percent_migration:
        :type percent_migration: float
        :param percent_selection:
        :type percent_selection: float
        :rtype: None
        :constraint: percent_cross and percent_mutation and percent_migration <=  1 - percent_selection
        """
        self._set_percent_cross(percent_cross)
        self._set_percent_mutation(percent_mutation)
        self._set_percent_migration(percent_migration)
        self._set_percent_selection(percent_selection)

    def get_n_samples(self) -> int:
        """get_n_samples.

        :rtype: int
        """
        return self._n_samples

    def get_n_jobs(self) -> int:
        """get_n_jobs.

        :rtype: int
        """
        return self._n_jobs

    def get_n_machines(self) -> int:
        """get_n_machines.

        :rtype: int
        """
        return self._n_machines

    def get_n_operations(self) -> int:
        """get_n_operations.

        :rtype: int
        """
        return self._n_operations

    def get_fitness_type(self) -> cp.core.core.ndarray:
        """get_fitness_type.

        :rtype: cp.core.core.ndarray
        """
        return self._fitness_type

    def get_percent_cross(self) -> float:
        """get_percent_cross.

        :rtype: float
        """
        return self._percent_cross

    def get_percent_intra_cross(self) -> float:
        """get_percent_intra_cross.

        :rtype: float
        """
        return self._percent_intra_cross

    def get_percent_mutation(self) -> float:
        """get_percent_mutation.

        :rtype: float
        """
        return self._percent_mutation

    def get_percent_intra_mutation(self) -> float:
        """get_percent_intra_mutation.

        :rtype: float
        """
        return self._percent_intra_mutation

    def get_percent_migration(self) -> float:
        """get_percent_migration.

        :rtype: float
        """
        return self._percent_migration

    def get_percent_selection(self) -> float:
        """get_percent_selection.

        :rtype: float
        """
        return self._percent_selection

    def get_fitness(self) -> cp.core.core.ndarray:
        """get_fitness.

        :rtype: cp.core.core.ndarray
        """
        return self._fitness

    def get_population(self) -> cp.core.core.ndarray:
        """get_population.

        :rtype: cp.core.core.ndarray
        """
        return self._population

    def get_processing_time(self) -> cp.core.core.ndarray:
        """get_processing_time.

        :rtype: cp.core.core.ndarray
        """
        return self._processing_time

    def get_machine_sequence(self) -> cp.core.core.ndarray:
        """get_machine_sequence.

        :rtype: cp.core.core.ndarray
        """
        return self._machine_sequence

    def get_due_date(self) -> cp.core.core.ndarray:
        """get_due_date.

        :rtype: cp.core.core.ndarray
        """
        return self._due_date

    def get_weights(self) -> cp.core.core.ndarray:
        """get_weights.

        :rtype: cp.core.core.ndarray
        """
        return self._weights

    def exec_permutationA0001(self) -> cp.core.core.ndarray:
        """exec_permutationA0001.

        :rtype: cp.core.core.ndarray
        """
        return self._permutationA0001(
            self.get_n_jobs(), self.get_n_operations(), self.get_n_samples()
        )

    def exec_crossA0001(self) -> None:
        """exec_crossA0001.

        :rtype: None
        """
        x_population = cp.copy(self.get_population())
        x_aux = cp.copy(self.get_population())
        index_selection = int(self.get_n_samples() * self.get_percent_selection())
        index_cross = int(self.get_n_samples() * self.get_percent_cross())
        cp.random.shuffle(x_population)
        if index_cross % 2 == 0:
            y_population = self._crossA0001(
                x_population[index_selection:, :][0:index_cross, :],
                self.get_n_jobs(),
                self.get_n_operations(),
                index_cross,
                self.get_percent_intra_cross(),
            )
            x_aux[index_selection:, :][-index_cross:, :] = y_population
            self.set_population(x_aux)
        else:
            index_cross -= 1
            y_population = self._crossA0001(
                x_population[index_selection:, :][0:index_cross, :],
                self.get_n_jobs(),
                self.get_n_operations(),
                self.get_n_samples(),
                self.get_percent_intra_cross(),
            )
            x_aux[index_selection:, :][-index_cross:, :] = y_population
            self.set_population(x_aux)

    def exec_mutationA0001(self) -> None:
        """exec_mutationA0001.

        :rtype: None
        """
        x_population = cp.copy(self.get_population())
        x_aux = cp.copy(self.get_population())
        index_selection = int(self.get_n_samples() * self.get_percent_selection())
        index_mutation = int(self.get_n_samples() * self.get_percent_mutation())
        cp.random.shuffle(x_population)
        y_population = self._mutationA0001(
            x_population[index_selection:, :][-index_mutation:, :],
            self.get_n_jobs(),
            self.get_n_operations(),
            index_mutation,
            self.get_percent_intra_mutation(),
        )
        x_aux[index_selection:, :][0:index_mutation, :] = y_population
        self.set_population(x_aux)

    def exec_migrationA0001(self) -> None:
        """exec_migrationA0001.

        :rtype: None
        """
        x_population = cp.copy(self.get_population())
        x_aux = cp.copy(self.get_population())
        index_selection = int(self.get_n_samples() * self.get_percent_selection())
        index_migration = int(self.get_n_samples() * self.get_percent_migration())
        cp.random.shuffle(x_population)
        y_population = self._permutationA0001(
            self.get_n_jobs(), self.get_n_operations(), index_migration
        )
        x_aux[index_selection:, :][-index_migration:, :] = y_population
        self.set_population(x_aux)

    def exec_sortA0001(self) -> None:
        """exec_sortA0001.

        :rtype: None
        """
        x_population = cp.copy(self.get_population())
        x_sort = cp.copy(self.get_fitness())
        y_population, y_sort = self._sortA0001(
            x_population,
            x_sort,
            self.get_n_jobs(),
            self.get_n_operations(),
            self.get_n_samples(),
        )
        self.set_population(y_population)
        self._set_fitness(y_sort)

    def exec_fitnessA0001(self) -> None:
        """exec_fitnessA0001.

        :rtype: None
        """
        x_population = cp.copy(self.get_population())
        fitness = self._fitnessA0001(
            x_population,
            self.get_due_date(),
            self.get_weights(),
            self.get_processing_time(),
            self.get_machine_sequence(),
            self.get_n_jobs(),
            self.get_n_samples(),
            self.get_n_machines(),
        )
        self._set_fitness(fitness[self.get_fitness_type()])

    def get_plan(self, row: int, fact_conv: int, start_time: int) -> list:
        """get_plan.

        :param row:
        :type row: int
        :param fact_conv:
        :type fact_conv: int
        :param start_time:
        :type start_time: int
        :rtype: dict
        """
        x_population = cp.copy(self.get_population())
        return self._get_planA0001(
            row,
            x_population,
            self.get_processing_time(),
            self.get_machine_sequence(),
            self.get_n_jobs(),
            self.get_n_machines(),
            fact_conv,
            start_time,
        )
