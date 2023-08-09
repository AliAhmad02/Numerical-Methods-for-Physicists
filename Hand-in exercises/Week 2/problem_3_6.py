"""Problem 3.6."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve


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


def solve_advection(
    start: float,
    end: float,
    N: int,
    D: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Solving the diffusion-advection equation."""
    x: NDArray[np.float64] = np.linspace(start, end, N)
    dx: NDArray[np.float64] = np.diff(x)[0]
    df2dt = 1 / dx**2 * upper_lower_diagonal(N, 1, -2, 1)
    dfdt = 1 / (2 * dx) * upper_lower_diagonal(N, 1, 0, -1)
    sin_mat: NDArray[np.float64] = np.diag(-np.sin(x))
    A: NDArray[np.float64] = D * df2dt - dfdt @ sin_mat
    A[0] = np.append([1], np.zeros(N - 1))
    A[-1] = np.append(np.zeros(N - 1), [1])
    b: NDArray[np.int64] = np.zeros(N)
    b[0] = 1
    sol = solve(A, b)
    return x, sol


start: int = 0
end: int = 25
N: int = 1000
D_arr: NDArray[np.float64] = np.array([0.5, 2, 15])
for D in D_arr:
    plt.plot(*solve_advection(start, end, N, D), label=f"D={D}")
    plt.xlabel("x", fontsize=15)
    plt.ylabel("f(x)", fontsize=15)
plt.legend()
plt.tight_layout()
plt.show()
