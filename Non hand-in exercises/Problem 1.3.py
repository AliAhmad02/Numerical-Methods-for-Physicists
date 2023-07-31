"""Problem 1.3."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def upper_lower_diagonal(
    N: int,
    upper_val: float,
    main_val: float,
    lower_val: float,
) -> ArrayLike:
    """Return matrix with specificed values on given diagonals."""
    upper_vals: ArrayLike = np.ones(N - 1) * upper_val
    main_vals: ArrayLike = np.ones(N) * main_val
    lower_vals: ArrayLike = np.ones(N - 1) * lower_val
    diag_mat: ArrayLike = np.diag(main_vals, k=0)
    upper_diag_mat: ArrayLike = np.diag(upper_vals, k=1)
    lower_diag_mat: ArrayLike = np.diag(lower_vals, k=-1)
    return diag_mat + lower_diag_mat + upper_diag_mat


A = upper_lower_diagonal(10, 0.5, 0, -0.5)
print(f"\n a) \n {A}")

B = upper_lower_diagonal(10, 1, -2, 1)
print(f"\n b) \n {B}")

B[0] = np.append([1], np.zeros(9))
B[-1] = np.append(np.zeros(9), [1])

print(f"\n c) \n {B}")
