"""Problem 4.2."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve


def laplacian(N: int, dx: float) -> NDArray[np.float64]:
    """Return the Laplacian of dimension N²."""
    D2: NDArray[np.int64] = np.eye(N, N, 1) + np.eye(N, N, -1) - 2 * np.eye(N)
    L: NDArray[np.float64] = np.kron(D2, np.eye(N)) + np.kron(np.eye(N), D2)
    return 1 / dx**2 * L


def enforce_boundary_condition(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    indices: NDArray[np.float64],
    boundary_val: float,
) -> None:
    """Enforcing boundary conditions on matrix A."""
    for idx in indices:
        A[idx] = np.zeros(len(A))
        A[idx, idx] = 1
        b[idx] = boundary_val


N: int = 50
dx: float = 0.1
linspace: NDArray[np.float64] = np.linspace(0, N * dx, N)
xs, ys = np.meshgrid(linspace, linspace)
x: NDArray[np.float64] = xs.flatten()
y: NDArray[np.float64] = ys.flatten()
L = laplacian(N, dx)
b: NDArray[np.float64] = np.zeros(N**2)
left: NDArray[np.float64] = np.where(x == linspace[0])[0]
right: NDArray[np.float64] = np.where(x == linspace[-1])[0]
lower: NDArray[np.float64] = np.where(y == linspace[0])[0]
upper: NDArray[np.float64] = np.where(y == linspace[-1])[0]
enforce_boundary_condition(L, b, left, 2)
enforce_boundary_condition(L, b, right, 1)
enforce_boundary_condition(L, b, lower, 0)
enforce_boundary_condition(L, b, upper, 1)
sol = solve(L, b)
plt.imshow(sol.reshape(N, N), origin="lower")
plt.show()
