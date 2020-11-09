from __future__ import division
from numba import cuda
import numpy as np
import cupy as cp
import math


class SortA0001:
    def __init__(self) -> None:
        pass

    def _sortA0001(
        self,
        X: cp.core.core.ndarray,
        y: cp.core.core.ndarray,
        digits: int,
        repetitions: int,
        n_samples: int,
    ) -> cp.core.core.ndarray:
        def sortAC0001() -> cp.core.core.ndarray:

            X_AUX = cp.copy(X)
            return X_AUX[y.argsort()], cp.sort(y)

        return sortAC0001()
