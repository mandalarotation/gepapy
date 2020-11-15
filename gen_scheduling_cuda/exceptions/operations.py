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
            if n_samples < 2:
                raise SetException(
                    "n_samples -> {} ; expected -> n_samples >= 2".format(
                        type(n_samples)
                    )
                )

            if type(n_samples) != int:
                raise SetException(
                    "type(n_samples) -> {} ; expected -> <class 'int'>".format(
                        type(n_samples)
                    )
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
            if type(n_jobs) != int:
                raise SetException(
                    "type(n_jobs) -> {} ; expected -> <class 'int'>".format(
                        type(n_jobs)
                    )
                )
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
            if type(n_machines) != int:
                raise SetException(
                    "type(n_machines) -> {} ; expected -> <class 'int'>".format(
                        type(n_machines)
                    )
                )
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
            if type(n_operations) != int:
                raise SetException(
                    "type(n_operations) -> {} ; expected -> <class 'int'>".format(
                        type(n_operations)
                    )
                )
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
            fitness_list = [
                "E_C",
                "E_L",
                "E_LT",
                "E_U",
                "E_Lw",
                "E_LTw",
                "E_Uw",
                "max_C",
            ]
            if not (fitness_type in fitness_list):
                raise SetException(
                    "type(fitness_type) -> {} ; expected -> ['E_C','E_L','E_LT','E_U','E_Lw','E_LTw','E_Uw','max_C']".format(  # noqa: E501
                        type(fitness_type)
                    )
                )
            if type(fitness_type) != str:
                raise SetException(
                    "type(fitness_type) -> {} ; expected -> <class 'str'>".format(
                        type(fitness_type)
                    )
                )
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
            if percent_cross <= 0 or percent_cross >= 1:
                raise SetException(
                    "type(percent_cross) -> {} ; expected -> Interval[(0,1)]".format(
                        type(percent_cross)
                    )
                )

            if type(percent_cross) != float:
                raise SetException(
                    "type(percent_cross) -> {} ; expected -> <class 'float'>".format(
                        type(percent_cross)
                    )
                )
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
            if percent_intra_cross <= 0 or percent_intra_cross >= 1:
                raise SetException(
                    "type(percent_intra_cross) -> {} ; expected -> Interval[(0,1)]".format(
                        type(percent_intra_cross)
                    )
                )

            if type(percent_intra_cross) != float:
                raise SetException(
                    "type(percent_intra_cross) -> {} ; expected -> <class 'float'>".format(
                        type(percent_intra_cross)
                    )
                )
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
            if percent_mutation <= 0 or percent_mutation >= 1:
                raise SetException(
                    "type(percent_mutation) -> {} ; expected -> Interval[(0,1)]".format(
                        type(percent_mutation)
                    )
                )
            if type(percent_mutation) != float:
                raise SetException(
                    "type(percent_mutation) -> {} ; expected -> <class 'float'>".format(
                        type(percent_mutation)
                    )
                )
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
            if percent_intra_mutation <= 0 or percent_intra_mutation >= 1:
                raise SetException(
                    "type(percent_intra_mutation) -> {} ; expected -> Interval[(0,1)]".format(
                        type(percent_intra_mutation)
                    )
                )
            if type(percent_intra_mutation) != float:
                raise SetException(
                    "type(percent_intra_mutation) -> {} ; expected -> <class 'float'>".format(
                        type(percent_intra_mutation)
                    )
                )
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
            if percent_migration <= 0 or percent_migration >= 1:
                raise SetException(
                    "type(percent_migration) -> {} ; expected -> Interval[(0,1)]".format(
                        type(percent_migration)
                    )
                )
            if type(percent_migration) != float:
                raise SetException(
                    "type(percent_migration) -> {} ; expected -> <class 'float'>".format(
                        type(percent_migration)
                    )
                )
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
            if percent_selection <= 0 or percent_selection >= 1:
                raise SetException(
                    "type(percent_selection) -> {} ; expected -> Interval[(0,1)]".format(
                        type(percent_selection)
                    )
                )
            if type(percent_selection) != float:
                raise SetException(
                    "type(percent_selection) -> {} ; expected -> <class 'float'>".format(
                        type(percent_selection)
                    )
                )
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
            if type(fitness) != cp.core.core.ndarray:
                raise SetException(
                    "type(fitness) -> {} ; expected -> <class 'cp.core.core.ndarray'>".format(
                        type(fitness)
                    )
                )
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
            if type(population) != cp.core.core.ndarray:
                raise SetException(
                    "type(population) -> {} ; expected -> <class 'cp.core.core.ndarray'>".format(
                        type(population)
                    )
                )
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
            if not (
                type(processing_time) == list
                or type(processing_time) == np.ndarray
                or type(processing_time) == cp.core.core.ndarray
            ):
                raise SetException(
                    "type(processing_time) -> {} ; expected -> <class 'Optional[Union[list, np.ndarray, cp.core.core.ndarray]]'>".format(  # noqa: E501
                        type(processing_time)
                    )
                )
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
            if not (
                type(machine_sequence) == list
                or type(machine_sequence) == np.ndarray
                or type(machine_sequence) == cp.core.core.ndarray
            ):
                raise SetException(
                    "type(machine_sequence) -> {} ; expected -> <class 'Optional[Union[list, np.ndarray, cp.core.core.ndarray]]'>".format(  # noqa: E501
                        type(machine_sequence)
                    )
                )
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
            if not (
                type(due_date) == list
                or type(due_date) == np.ndarray
                or type(due_date) == cp.core.core.ndarray
            ):
                raise SetException(
                    "type(due_date) -> {} ; expected -> <class 'Optional[Union[list, np.ndarray, cp.core.core.ndarray]]'>".format(  # noqa: E501
                        type(due_date)
                    )
                )
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
            if not (
                type(weights) == list
                or type(weights) == np.ndarray
                or type(weights) == cp.core.core.ndarray
            ):
                raise SetException(
                    "type(weights) -> {} ; expected -> <class 'Optional[Union[list, np.ndarray, cp.core.core.ndarray]]'>".format(  # noqa: E501
                        type(weights)
                    )
                )
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
            if (
                percent_cross > percent_selection
                or percent_mutation > percent_selection
                or percent_migration > percent_selection
            ):
                raise SetException(
                    "False ; expected ->  True  where percent_cross < percent_selection and percent_mutation < percent_selection and percent_migration < percent_selection"  # noqa: E501
                )
            return f(
                self,
                percent_cross,
                percent_mutation,
                percent_migration,
                percent_selection,
            )

        return cast(TFun, wrapper)
