from __future__ import division
from numba import cuda
import numpy as np
import cupy as cp
import math


class FitnessSM0001:
    def __init__(self) -> None:
        pass

    def _fitnessSM0001(
        self,
        X: cp.core.core.ndarray,
        p: cp.core.core.ndarray,
        d: cp.core.core.ndarray,
        w: cp.core.core.ndarray,
        digits: int,
        n_samples: int,
    ) -> cp.core.core.ndarray:
        def fitnessSMC0001() -> cp.core.core.ndarray:
            @cuda.jit
            def kernel(
                X: cp.core.core.ndarray,
                y: cp.core.core.ndarray,
                p: cp.core.core.ndarray,
                d: cp.core.core.ndarray,
                w: cp.core.core.ndarray,
            ) -> None:
                row = cuda.grid(1)
                if row < n_samples:
                    ptime = 0
                    tardiness = 0
                    for j in range(digits):
                        ptime = ptime + p[row, int((X[row][j]))]
                        tardiness = tardiness + w[row, int((X[row][j]))] * max(
                            ptime - d[row, int((X[row][j]))], 0
                        )
                        y[row] = tardiness

            y = cp.zeros(n_samples, dtype=cp.float32)
            threadsperblock = 16
            blockspergrid_x = int(math.ceil(n_samples / threadsperblock))
            blockspergrid = blockspergrid_x

            cuda.synchronize()
            kernel[blockspergrid, threadsperblock](X, y, p, d, w)
            cuda.synchronize()

            return y

        return fitnessSMC0001()
