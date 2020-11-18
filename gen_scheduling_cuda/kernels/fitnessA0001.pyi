import cupy as cp  # type: ignore

class FitnessA0001:
    def __init__(self) -> None: ...
    def _fitnessA0001(
        self,
        x: cp.core.core.ndarray,
        d: cp.core.core.ndarray,
        w: cp.core.core.ndarray,
        T: cp.core.core.ndarray,
        M: cp.core.core.ndarray,
        digits: int,
        n_samples: int,
        n_machines: int,
    ) -> cp.core.core.ndarray: ...
    def _get_planA0001(
        self,
        row: int,
        X: cp.core.core.ndarray,
        T: cp.core.core.ndarray,
        M: cp.core.core.ndarray,
        digits: int,
        n_machines: int,
        fact_conv: float,
        start_time: int,
    ) -> list: ...
