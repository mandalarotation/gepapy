from __future__ import division
from numba import cuda
import numpy as np
import cupy as cp
import math


class MutationA0001:
    def __init__(self) -> None:
        pass

    def _mutationA0001(
        self,
        X: cp.core.core.ndarray,
        digits: int,
        repetitions: int,
        n_samples: int,
        percent_m: float,
    ):
        def mutationAC0001() -> cp.core.core.ndarray:
            @cuda.jit
            def kernel(
                X_AUX: cp.core.core.ndarray,
                AL: cp.core.core.ndarray,
                digits: int,
                repetitions: int,
                n_samples: int,
            ):
                row = cuda.grid(1)
                if row < n_samples:
                    for i in range(AL.shape[1]):
                        x_aux = X_AUX[row, int(math.ceil(AL[row, i, 0]))]
                        X_AUX[row, int(math.ceil(AL[row, i, 0]))] = X_AUX[
                            row, int(math.ceil(AL[row, i, 1]))
                        ]
                        X_AUX[row, int(math.ceil(AL[row, i, 1]))] = x_aux

            x_dim_1 = digits * repetitions

            X_AUX = cp.copy(X)

            AL = cp.array(
                cp.random.rand(n_samples, int(x_dim_1 * percent_m), 2), dtype=cp.float32
            ) * (x_dim_1 - 1)

            threadsperblock = 16
            blockspergrid_x = int(math.ceil((n_samples) / threadsperblock))
            blockspergrid = blockspergrid_x

            cuda.synchronize()
            kernel[blockspergrid, threadsperblock](
                X_AUX, AL, digits, repetitions, n_samples
            )
            cuda.synchronize()

            return X_AUX

        return mutationAC0001()
