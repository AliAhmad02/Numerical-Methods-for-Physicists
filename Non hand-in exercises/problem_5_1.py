"""Problem 5.1."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def bak_sneppen(
    N: int,
    time_steps: int,
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    """Implement Bak-Sneppen method for given N and time steps."""
    f: NDArray[np.float64] = np.random.uniform(size=N)
    f_mins = []
    age: NDArray[np.int64] = np.zeros(N)
    ages = []
    for i in range(time_steps):
        minimum_idx: int = np.argmin(f)
        f_mins.append(f[minimum_idx])
        f[minimum_idx] = np.random.uniform()
        age += 1
        age[minimum_idx] = 0
        if minimum_idx == N - 1:
            f[minimum_idx - 1] = np.random.uniform()
            f[0] = np.random.uniform()
            age[[0, minimum_idx - 1]] = 0

        elif minimum_idx == 0:
            f[minimum_idx + 1] = np.random.uniform()
            f[N - 1] = np.random.uniform()
            age[[minimum_idx + 1, N - 1]] = 0
        else:
            f[minimum_idx + 1] = np.random.uniform()
            f[minimum_idx - 1] = np.random.uniform()
            age[[minimum_idx + 1, minimum_idx - 1]] = 0
        if i % 100 == 0:
            ages.append(age.copy())
    return np.array(f_mins), ages


bs_100k, ages_100k = bak_sneppen(1000, 100_000)
t: NDArray[np.int64] = np.arange(100_000)
N_array: NDArray[np.int64] = np.arange(1000)
t1: NDArray[np.int64] = np.arange(0, 100_000, 100)
xs, ys = np.meshgrid(N_array, t1)

plt.figure(figsize=(6, 4))
plt.plot(t, bs_100k)
plt.xlabel("Time")
plt.ylabel("Minimum Fitness")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.pcolormesh(xs, ys, ages_100k, cmap="gist_rainbow")
plt.xlabel("Grid point", fontsize=15)
plt.ylabel("Time", fontsize=15)
plt.title("Age of each species", fontsize=15)
plt.colorbar()
plt.tight_layout()
plt.show()
