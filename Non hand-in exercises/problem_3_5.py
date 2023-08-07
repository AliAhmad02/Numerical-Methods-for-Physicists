"""Problem 3.5."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from problem_1_3 import upper_lower_diagonal
from scipy.linalg import solve

dx: float = 0.01
start: int = -5
end: int = 5
N: int = int((end - start) / dx) + 1
x: NDArray[np.float64] = np.linspace(start, end, N)

dx2dt: NDArray[np.float64] = 1 / dx**2 * upper_lower_diagonal(N, 1, -2, 1)

exp_mat: NDArray[np.float64] = np.diag(np.exp(-(x**2)))
A: NDArray[np.float64] = dx2dt + exp_mat
b: NDArray[np.float64] = np.zeros(N)

A[0] = np.append([1], np.zeros(N - 1))
A[-1] = 1 / dx * np.append(np.zeros(N - 3), [1 / 2, -2, 3 / 2])
b[0] = 1
sol: NDArray[np.float64] = solve(A, b)

dx2dt1 = np.copy(dx2dt)
dx2dt1[0] = np.append([1], np.zeros(N - 1))
dx2dt1[-1] = 1 / dx * np.append(np.zeros(N - 3), [1 / 2, -2, 3 / 2])
b1 = -np.exp(-(x**2))
b1[0] = 1
b1[-1] = 0
sol1 = solve(dx2dt1, b1)

print(
    np.allclose((A - np.diag(np.exp(-(x**2))))[1:-1], dx2dt1[1:-1]),
)

plt.plot(x, sol)
plt.plot(x, sol1)
plt.xlabel("x", fontsize=15)
plt.ylabel("f(x)", fontsize=15)
plt.show()
