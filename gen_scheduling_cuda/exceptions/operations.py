import cupy as cp  # type: ignore
import numpy as np  # type: ignore
from functools import wraps
from typing import TypeVar, Callable, Any, cast, Optional, Union

TFun = TypeVar("TFun", bound=Callable[..., Any])


class SetException(Exception):
    """MyOperatorException."""

    def __init__(self, message: str) -> None:
        """__init__.

        :param message:
        :type message: str
        :rtype: None
        """
        self._message: str = self.set_message(message)

    def set_message(self, message: str) -> str:
        """set_message.

        :param message:
        :type message: str
        :rtype: str
        """
        return message

    def get_message(self) -> str:
        """get_message.

        :rtype: str
        """
        return self._message

    def __str__(self) -> str:
        """__str__.

        :rtype: str
        """
        return self.get_message()


class Check:
    def __inti__(self) -> None:
        """__inti__.

        :rtype: None
        """
        pass

    @classmethod
    def _set_n_samples_check(self, f: TFun) -> TFun:
        """_set_n_samples_check.

        :param f:
        :type f: TFun
        :rtype: TFun
        """

        @wraps(f)
        def wrapper(self, n_samples: int) -> Optional[Union[None, int]]:
            """wrapper.

            :param n_samples:
            :type n_samples: int
            :rtype: Optional[Union[None, int]]
            """
            if type(n_samples) != int:
                raise SetException(
                    "type(n_samples) -> {} ; expected -> int".format(type(n_samples))
                )
            return f(self, n_samples)

        return cast(TFun, wrapper)

    @classmethod
    def _set_n_jobs_check(self, f: TFun) -> TFun:
        """_set_n_jobs_check.

        :param f:
        :type f: TFun
        :rtype: TFun
        """

        @wraps(f)
        def wrapper(self, n_jobs: int) -> Optional[Union[None, int]]:
            """wrapper.

            :param n_jobs:
            :type n_jobs: int
            :rtype: Optional[Union[None, int]]
            """
            print("estoy decorada")
            return f(self, n_jobs)

        return cast(TFun, wrapper)

    @classmethod
    def _set_n_machines_check(self, f: TFun) -> TFun:
        """_set_n_machines_check.

        :param f:
        :type f: TFun
        :rtype: TFun
        """

        @wraps(f)
        def wrapper(self, n_machines: int) -> Optional[Union[None, int]]:
            """wrapper.

            :param n_machines:
            :type n_machines: int
            :rtype: Optional[Union[None, int]]
            """
            print("estoy decorada")
            return f(self, n_machines)

        return cast(TFun, wrapper)

    @classmethod
    def _set_n_operations_check(self, f: TFun) -> TFun:
        """_set_n_operations_check.

        :param f:
        :type f: TFun
        :rtype: TFun
        """

        @wraps(f)
        def wrapper(self, n_operations: int) -> Optional[Union[None, int]]:
            """wrapper.

            :param n_operations:
            :type n_operations: int
            :rtype: Optional[Union[None, int]]
            """
            print("estoy decorada")
            return f(self, n_operations)

        return cast(TFun, wrapper)

    @classmethod
    def _set_fitness_type_check(self, f: TFun) -> TFun:
        """_set_fitness_type_check.

        :param f:
        :type f: TFun
        :rtype: TFun
        """

        @wraps(f)
        def wrapper(self, fitness_type: str) -> Optional[Union[None, str]]:
            """wrapper.

            :param fitness_type:
            :type fitness_type: str
            :rtype: Optional[Union[None, str]]
            """
            print("estoy decorada")
            return f(self, fitness_type)

        return cast(TFun, wrapper)

    @classmethod
    def _set_percent_cross_check(self, f: TFun) -> TFun:
        """_set_percent_cross_check.

        :param f:
        :type f: TFun
        :rtype: TFun
        """

        @wraps(f)
        def wrapper(self, percent_cross: float) -> Optional[Union[None, float]]:
            """wrapper.

            :param percent_cross:
            :type percent_cross: float
            :rtype: Optional[Union[None, float]]
            """
            print("estoy decorada")
            return f(self, percent_cross)

        return cast(TFun, wrapper)

    @classmethod
    def _set_percent_intra_cross_check(self, f: TFun) -> TFun:
        """_set_percent_intra_cross_check.

        :param f:
        :type f: TFun
        :rtype: TFun
        """

        @wraps(f)
        def wrapper(self, percent_intra_cross: float) -> Optional[Union[None, float]]:
            """wrapper.

            :param percent_intra_cross:
            :type percent_intra_cross: float
            :rtype: Optional[Union[None, float]]
            """
            print("estoy decorada")
            return f(self, percent_intra_cross)

        return cast(TFun, wrapper)

    @classmethod
    def _set_percent_mutation_check(self, f: TFun) -> TFun:
        """_set_percent_mutation_check.

        :param f:
        :type f: TFun
        :rtype: TFun
        """

        @wraps(f)
        def wrapper(self, percent_mutation: float) -> Optional[Union[None, float]]:
            """wrapper.

            :param percent_mutation:
            :type percent_mutation: float
            :rtype: Optional[Union[None, float]]
            """
            print("estoy decorada")
            return f(self, percent_mutation)

        return cast(TFun, wrapper)

    @classmethod
    def _set_percent_intra_mutation_check(self, f: TFun) -> TFun:
        """_set_percent_intra_mutation_check.

        :param f:
        :type f: TFun
        :rtype: TFun
        """

        @wraps(f)
        def wrapper(
            self, percent_intra_mutation: float
        ) -> Optional[Union[None, float]]:
            """wrapper.

            :param percent_intra_mutation:
            :type percent_intra_mutation: float
            :rtype: Optional[Union[None, float]]
            """
            print("estoy decorada")
            return f(self, percent_intra_mutation)

        return cast(TFun, wrapper)

    @classmethod
    def _set_percent_migration_check(self, f: TFun) -> TFun:
        """_set_percent_migration_check.

        :param f:
        :type f: TFun
        :rtype: TFun
        """

        @wraps(f)
        def wrapper(self, percent_migration: float) -> Optional[Union[None, float]]:
            """wrapper.

            :param percent_migration:
            :type percent_migration: float
            :rtype: Optional[Union[None, float]]
            """
            print("estoy decorada")
            return f(self, percent_migration)

        return cast(TFun, wrapper)

    @classmethod
    def _set_percent_selection_check(self, f: TFun) -> TFun:
        """_set_percent_selection_check.

        :param f:
        :type f: TFun
        :rtype: TFun
        """

        @wraps(f)
        def wrapper(self, percent_selection: float) -> Optional[Union[None, float]]:
            """wrapper.

            :param percent_selection:
            :type percent_selection: float
            :rtype: Optional[Union[None, float]]
            """
            print("estoy decorada")
            return f(self, percent_selection)

        return cast(TFun, wrapper)

    @classmethod
    def _set_fitness_check(self, f: TFun) -> TFun:
        """_set_fitness_check.

        :param f:
        :type f: TFun
        :rtype: TFun
        """

        @wraps(f)
        def wrapper(
            self,
            fitness: cp.core.core.ndarray,
        ) -> Optional[Union[None, cp.core.core.ndarray]]:
            """wrapper.

            :param fitness:
            :type fitness: cp.core.core.ndarray
            :rtype: Optional[Union[None, cp.core.core.ndarray]]
            """
            print("estoy decorada")
            return f(self, fitness)

        return cast(TFun, wrapper)

    @classmethod
    def _set_population_check(self, f: TFun) -> TFun:
        """_set_population_check.

        :param f:
        :type f: TFun
        :rtype: TFun
        """

        @wraps(f)
        def wrapper(
            self, population: Optional[Union[cp.core.core.ndarray, None]] = None
        ) -> Optional[Union[None, cp.core.core.ndarray]]:
            """wrapper.

            :param population:
            :type population: Optional[Union[cp.core.core.ndarray, None]]
            :rtype: Optional[Union[None, cp.core.core.ndarray]]
            """
            print("estoy decorada")
            return f(self, population)

        return cast(TFun, wrapper)

    @classmethod
    def _set_processing_time_check(self, f: TFun) -> TFun:
        """_set_processing_time_check.

        :param f:
        :type f: TFun
        :rtype: TFun
        """

        @wraps(f)
        def wrapper(
            self,
            processing_time: Optional[Union[list, np.ndarray, cp.core.core.ndarray]],
        ) -> cp.core.core.ndarray:
            """wrapper.

            :param processing_time:
            :type processing_time: Optional[Union[list, np.ndarray, cp.core.core.ndarray]]
            :rtype: cp.core.core.ndarray
            """
            print("estoy decorada")
            return f(self, processing_time)

        return cast(TFun, wrapper)

    @classmethod
    def _set_machine_sequence_check(self, f: TFun) -> TFun:
        """_set_machine_sequence_check.

        :param f:
        :type f: TFun
        :rtype: TFun
        """

        @wraps(f)
        def wrapper(
            self,
            machine_sequence: Optional[Union[list, np.ndarray, cp.core.core.ndarray]],
        ) -> cp.core.core.ndarray:
            """wrapper.

            :param machine_sequence:
            :type machine_sequence: Optional[Union[list, np.ndarray, cp.core.core.ndarray]]
            :rtype: cp.core.core.ndarray
            """
            print("estoy decorada")
            return f(self, machine_sequence)

        return cast(TFun, wrapper)

    @classmethod
    def _set_due_date_check(self, f: TFun) -> TFun:
        """_set_due_date_check.

        :param f:
        :type f: TFun
        :rtype: TFun
        """

        @wraps(f)
        def wrapper(
            self, due_date: Optional[Union[list, np.ndarray, cp.core.core.ndarray]]
        ) -> cp.core.core.ndarray:
            """wrapper.

            :param due_date:
            :type due_date: Optional[Union[list, np.ndarray, cp.core.core.ndarray]]
            :rtype: cp.core.core.ndarray
            """
            print("estoy decorada")
            return f(self, due_date)

        return cast(TFun, wrapper)

    @classmethod
    def _set_weights_check(self, f: TFun) -> TFun:
        """_set_weights_check.

        :param f:
        :type f: TFun
        :rtype: TFun
        """

        @wraps(f)
        def wrapper(
            self, weights: Optional[Union[list, np.ndarray, cp.core.core.ndarray]]
        ) -> cp.core.core.ndarray:
            """wrapper.

            :param weights:
            :type weights: Optional[Union[list, np.ndarray, cp.core.core.ndarray]]
            :rtype: cp.core.core.ndarray
            """
            print("estoy decorada")
            return f(self, weights)

        return cast(TFun, wrapper)

    @classmethod
    def _set_percents_c_m_m_s_check(self, f: TFun) -> TFun:
        """_set_percent_mutation_check.

        :param f:
        :type f: TFun
        :rtype: TFun
        """

        @wraps(f)
        def wrapper(
            self,
            percent_cross: float,
            percent_mutation: float,
            percent_migration: float,
            percent_selection: float,
        ) -> None:
            """wrapper.

            :param percent_cross:
            :type percent_cross: float
            :param percent_mutation:
            :type percent_mutation: float
            :param percent_migration:
            :type percent_migration: float
            :param percent_selection:
            :type percent_selection: float
            :rtype: None
            """
            print("estoy decorada")
            return f(
                self,
                percent_cross,
                percent_mutation,
                percent_migration,
                percent_selection,
            )

        return cast(TFun, wrapper)
