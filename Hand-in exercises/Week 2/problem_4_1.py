"""Problem 4.1."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from problem_3_6 import upper_lower_diagonal
from scipy.linalg import solve

start: int = 0
end: int = 1
tend: int = 3
N: int = 1000
x = np.linspace(start, end, N)
dx: float = np.diff(x)[0]
dt: float = 0.05
Nt: int = int(tend / dt) + 1
Nplot: int = 7
df2dt = 1 / dx**2 * upper_lower_diagonal(N, 1, -2, 1)
t_plot: NDArray[np.float64] = np.linspace(start, tend, Nplot)
t_arr: NDArray[np.float64] = np.linspace(start, tend, Nt)
I: NDArray[np.int64] = np.identity(N)
A: NDArray[np.float64] = I - df2dt * dt
A[0] = np.append([1], np.zeros(N - 1))
A[-1] = (
    1
    / dx
    * np.append(
        np.zeros(N - 3),
        [1 / 2, -2, 3 / 2],
    )
)
b: NDArray[np.float64] = np.exp(-5 * x)
b[0] = 1
b[-1] = 0
for t in t_arr:
    sol = solve(A, b)
    b = sol
    b[0] = 1
    b[-1] = 0
    if t in t_plot:
        plt.plot(x[:-1], sol[:-1], label=f"t={t}")
plt.xlabel("x", fontsize=15)
plt.ylabel("f(x)", fontsize=15)
plt.legend()
plt.tight_layout()
plt.show()
