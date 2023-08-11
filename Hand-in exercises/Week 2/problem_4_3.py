"""Problem 4.3."""
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


A: NDArray[np.int64] = np.array([-1, -1, -1])
B: NDArray[np.int64] = np.array([1, 1, -1])
C: NDArray[np.int64] = np.array([-1, 1, 1])
D: NDArray[np.int64] = np.array([1, -1, 1])
start: int = -1
end: int = 1
N: int = 50
linspace: NDArray[np.float64] = np.linspace(start, end, N)
dx: float = np.diff(linspace)[0]
xs, ys = np.meshgrid(linspace, linspace)
x = xs.flatten()
y = ys.flatten()
L = laplacian(N, dx)
b: NDArray[np.float64] = np.zeros(N**2)
AC = np.where(x == -1)[0]
AD = np.where(y == -1)[0]
BC = np.where(y == 1)[0]
BD = np.where(x == 1)[0]
for idx in AC:
    L[idx] = np.zeros(N**2)
    L[idx, idx] = 1
    b[idx] = y[idx]
for idx in AD:
    L[idx] = np.zeros(N**2)
    L[idx, idx] = 1
    b[idx] = x[idx]
for idx in BC:
    L[idx] = np.zeros(N**2)
    L[idx, idx] = 1
    b[idx] = -x[idx]
for idx in BD:
    L[idx] = np.zeros(N**2)
    L[idx, idx] = 1
    b[idx] = -y[idx]
sol = solve(L, b)
fig = plt.figure(figsize=(7, 7), dpi=300)
ax = fig.add_subplot(projection="3d")
ax.plot(*zip(A, C), "o-", color="k", lw=3)
ax.plot(*zip(A, D), "o-", color="k", lw=3)
ax.plot(*zip(B, C), "o-", color="k", lw=3)
ax.plot(*zip(B, D), "o-", color="k", lw=3)
ax.plot_surface(xs, ys, sol.reshape(N, N), antialiased=False, color="yellow")
ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("y", fontsize=15)
ax.set_zlabel("z", fontsize=15)
plt.tight_layout()
plt.show()
