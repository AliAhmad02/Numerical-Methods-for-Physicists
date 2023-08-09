"""Problem 1.3."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def upper_lower_diagonal(
    N: int,
    upper_val: float,
    main_val: float,
    lower_val: float,
) -> NDArray[np.float64]:
    """Return matrix with specificed values on given diagonals."""
    upper_vals: NDArray[np.float64] = np.ones(N - 1) * upper_val
    main_vals: NDArray[np.float64] = np.ones(N) * main_val
    lower_vals: NDArray[np.float64] = np.ones(N - 1) * lower_val
    diag_mat: NDArray[np.float64] = np.diag(main_vals, k=0)
    upper_diag_mat: NDArray[np.float64] = np.diag(upper_vals, k=1)
    lower_diag_mat: NDArray[np.float64] = np.diag(lower_vals, k=-1)
    return diag_mat + lower_diag_mat + upper_diag_mat


if __name__ == "__main__":
    A = upper_lower_diagonal(10, 0.5, 0, -0.5)
    print(f"\n a) \n {A}")

    B = upper_lower_diagonal(10, 1, -2, 1)
    print(f"\n b) \n {B}")

    B[0] = np.append([1], np.zeros(9))
    B[-1] = np.append(np.zeros(9), [1])

    print(f"\n c) \n {B}")

    N: int = 5
    x: NDArray[np.float64] = np.zeros(N**2)
    y: NDArray[np.float64] = np.zeros(N**2)
    for i in range(N):
        for j in range(N):
            k = N * i + j
            x[k] = j
            y[k] = i
    print(f"\n d) x: \n {x} \n y: \n {y}")

    N1: int = 4
    L: NDArray[np.float64] = np.zeros((N1**2, N1**2))
    for k in range(N1**2):
        for l in range(N1**2):  # noqa: E741
            if k == l:
                L[k, l] = -4
            elif np.abs(x[k] - x[l]) == 1 and y[k] == y[l]:
                L[k, l] = 1
            elif np.abs(y[k] - y[l]) == 1 and x[k] == x[l]:
                L[k, l] = 1
    print(f"\n e) \n L: \n {L}")

    n: int = 4
    D1: NDArray[np.float64] = 0.5 * np.eye(n, n, 1) - 0.5 * np.eye(n, n, -1)
    D2: NDArray[np.float64] = np.eye(n, k=1) + np.eye(n, k=-1) - 2 * np.eye(n)
    xx, yy = np.meshgrid(np.arange(n), np.arange(n))
    x = xx.flatten()
    y = yy.flatten()

    L1: NDArray[np.float64] = np.kron(D2, np.eye(n)) + np.kron(np.eye(n), D2)
    print(f"\n h) \n L: \n {L1}")
