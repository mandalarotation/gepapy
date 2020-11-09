from __future__ import division
from numba import cuda
import numpy as np
import cupy as cp
import math


class PermutationA0001:
    def __init__(self) -> None:
        pass

    def _permutationA0001(
        self, digits: int, repetitions: int, n_samples: int
    ) -> cp.core.core.ndarray:
        def permutationAC0001() -> cp.core.core.ndarray:
            @cuda.jit
            def kernel(X: cp.core.core.ndarray, AL: cp.core.core.ndarray) -> None:
                row = cuda.grid(1)
                if row < X.shape[0]:
                    for j in range(0, X.shape[1], 1):
                        tmp = X[row, int(AL[row, j] * j + 0.5)]
                        X[row, int(AL[row, j] * j + 0.5)] = X[row, j]
                        X[row, j] = tmp
                cuda.syncthreads()

            X = cp.repeat(
                cp.repeat(
                    cp.expand_dims(cp.arange(digits, dtype=cp.float32), axis=0),
                    n_samples,
                    axis=0,
                ),
                repetitions,
                axis=1,
            )
            AL = cp.array(
                cp.random.rand(n_samples, digits * repetitions), dtype=cp.float32
            )

            threadsperblock = 16
            blockspergrid_x = int(math.ceil(n_samples / threadsperblock))
            blockspergrid = blockspergrid_x

            cuda.synchronize()
            kernel[blockspergrid, threadsperblock](X, AL)
            cuda.synchronize()

            AL = None

            return X

        return permutationAC0001()
