from __future__ import division
import cupy as cp  # type: ignore


class SortA0001:
    """SortA0001."""

    def __init__(self) -> None:
        """__init__.

        :rtype: None
        """
        pass

    def _sortA0001(
        self,
        X: cp.core.core.ndarray,
        y: cp.core.core.ndarray,
        digits: int,
        repetitions: int,
        n_samples: int,
    ) -> cp.core.core.ndarray:
        """_sortA0001.

        :param X:
        :type X: cp.core.core.ndarray
        :param y:
        :type y: cp.core.core.ndarray
        :param digits:
        :type digits: int
        :param repetitions:
        :type repetitions: int
        :param n_samples:
        :type n_samples: int
        :rtype: cp.core.core.ndarray
        """

        def sortAC0001() -> cp.core.core.ndarray:
            """sortAC0001.

            :rtype: cp.core.core.ndarray
            """

            X_AUX = cp.copy(X)
            return X_AUX[y.argsort()], cp.sort(y)

        return sortAC0001()
