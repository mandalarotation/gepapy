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
        pass

    def _set_n_samples_check(self, f: TFun) -> TFun:
        @wraps(f)
        def wrapper(n_samples: int) -> Optional[Union[None, int]]:
            print("estoy decorada")
            return f(n_samples)

        return cast(TFun, wrapper)

    def _set_n_jobs_check(self, f: TFun) -> TFun:
        @wraps(f)
        def wrapper(n_jobs: int) -> Optional[Union[None, int]]:
            print("estoy decorada")
            return f(n_jobs)

        return cast(TFun, wrapper)

    def _set_n_machines_check(self, f: TFun) -> TFun:
        @wraps(f)
        def wrapper(n_machines: int) -> Optional[Union[None, int]]:
            print("estoy decorada")
            return f(n_machines)

        return cast(TFun, wrapper)

    def _set_n_operations_check(self, f: TFun) -> TFun:
        @wraps(f)
        def wrapper(n_operations: int) -> Optional[Union[None, int]]:
            print("estoy decorada")
            return f(n_operations)

        return cast(TFun, wrapper)

    def _set_fitness_type_check(self, f: TFun) -> TFun:
        @wraps(f)
        def wrapper(fitness_type: str) -> Optional[Union[None, str]]:
            print("estoy decorada")
            return f(fitness_type)

        return cast(TFun, wrapper)

    def _set_percent_cross_check(self, f: TFun) -> TFun:
        @wraps(f)
        def wrapper(percent_cross: float) -> Optional[Union[None, float]]:
            print("estoy decorada")
            return f(percent_cross)

        return cast(TFun, wrapper)

    def _set_percent_intra_cross_check(self, f: TFun) -> TFun:
        @wraps(f)
        def wrapper(percent_intra_cross: float) -> Optional[Union[None, float]]:
            print("estoy decorada")
            return f(percent_intra_cross)

        return cast(TFun, wrapper)

    def _set_percent_mutation_check(self, f: TFun) -> TFun:
        @wraps(f)
        def wrapper(percent_mutation: float) -> Optional[Union[None, float]]:
            print("estoy decorada")
            return f(percent_mutation)

        return cast(TFun, wrapper)

    def _set_percent_intra_mutation_check(self, f: TFun) -> TFun:
        @wraps(f)
        def wrapper(percent_intra_mutation: float) -> Optional[Union[None, float]]:
            print("estoy decorada")
            return f(percent_intra_mutation)

        return cast(TFun, wrapper)

    def _set_percent_migration_check(self, f: TFun) -> TFun:
        @wraps(f)
        def wrapper(percent_migration: float) -> Optional[Union[None, float]]:
            print("estoy decorada")
            return f(percent_migration)

        return cast(TFun, wrapper)

    def _set_percent_selection_check(self, f: TFun) -> TFun:
        @wraps(f)
        def wrapper(percent_selection: float) -> Optional[Union[None, float]]:
            print("estoy decorada")
            return f(percent_selection)

        return cast(TFun, wrapper)

    def _set_fitness_check(self, f: TFun) -> TFun:
        @wraps(f)
        def wrapper(
            fitness: cp.core.core.ndarray,
        ) -> Optional[Union[None, cp.core.core.ndarray]]:
            print("estoy decorada")
            return f(fitness)

        return cast(TFun, wrapper)

    def _set_population_check(self, f: TFun) -> TFun:
        @wraps(f)
        def wrapper(
            population: Optional[Union[cp.core.core.ndarray, None]] = None
        ) -> Optional[Union[None, cp.core.core.ndarray]]:
            print("estoy decorada")
            return f(population)

        return cast(TFun, wrapper)

    def _set_processing_time_check(self, f: TFun) -> TFun:
        @wraps(f)
        def wrapper(
            processing_time: Optional[Union[list, np.ndarray, cp.core.core.ndarray]]
        ) -> cp.core.core.ndarray:
            print("estoy decorada")
            return f(processing_time)

        return cast(TFun, wrapper)

    def _set_machine_sequence_check(self, f: TFun) -> TFun:
        @wraps(f)
        def wrapper(
            machine_sequence: Optional[Union[list, np.ndarray, cp.core.core.ndarray]]
        ) -> cp.core.core.ndarray:
            print("estoy decorada")
            return f(machine_sequence)

        return cast(TFun, wrapper)

    def _set_due_date_check(self, f: TFun) -> TFun:
        @wraps(f)
        def wrapper(
            due_date: Optional[Union[list, np.ndarray, cp.core.core.ndarray]]
        ) -> cp.core.core.ndarray:
            print("estoy decorada")
            return f(due_date)

        return cast(TFun, wrapper)

    def _set_weights_check(self, f: TFun) -> TFun:
        @wraps(f)
        def wrapper(
            weights: Optional[Union[list, np.ndarray, cp.core.core.ndarray]]
        ) -> cp.core.core.ndarray:
            print("estoy decorada")
            return f(weights)

        return cast(TFun, wrapper)
