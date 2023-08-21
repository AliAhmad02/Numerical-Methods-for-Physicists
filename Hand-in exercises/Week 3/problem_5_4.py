"""Problem 5.4."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from numpy.typing import NDArray
from tqdm import tqdm


@njit
def energy_step(S, N, i, j):
    """Do a single step of the energy calculation.

    Modulo operator enforces periodic boundary conditions.
    """
    return S[i, j] * (
        S[(i + 1) % N, j]
        + S[(i - 1) % N, j]
        + S[i, (j + 1) % N]
        + S[i, (j - 1) % N]  # noqa: E501
    )


@njit
def ising_hamiltonian(S: NDArray[np.float64]) -> float:
    """Take an NxN spin matrix S and return its Hamiltonian."""
    N: int = len(S)
    H = 0
    for i in range(N):
        for j in range(N):
            H += energy_step(S, N, i, j)
    return -H / 2


@njit
def magnetization(S: NDArray[np.float64]) -> float:
    """Calculate the magnetization of a spin configuration S."""
    return np.mean(S)


@njit
def deltaE(S1: NDArray[np.float64], N, i, j) -> float:
    """Calculate difference of energy of two spin configurations."""
    return 2 * energy_step(S1, N, i, j)


@njit
def accept_reject_spin(
    S1: NDArray[np.float64],
    S2: NDArray[np.float64],
    energy_change: float,
    T: float,
) -> NDArray[np.float64]:
    """Accept or reject a spin flip based on change in energy."""
    alpha = min(1, np.exp(-energy_change / T))
    U = np.random.uniform(0, 1)
    S = (U <= alpha) * S2 + (U > alpha) * S1
    return S


@njit
def metropolis_hastings(
    S1: NDArray[np.int64],
    N: int,
    T: float,
) -> tuple[list[int], list[int]]:
    """Initialize random S and perform 1000 N² Metropolis Hastings steps."""
    iterations: int = 1000 * N**2
    iterations_save: list = []
    save_iter = [
        0,
        int((1 / 1000) * iterations),
        int((1 / 100) * iterations),
        int((1 / 50) * iterations),
        int((1 / 20) * iterations),
        int(iterations) - 1,
    ]
    S_save: list[NDArray[np.int64]] = []
    for i in range(iterations):
        S2: NDArray[np.float64] = S1.copy()
        x_idx, y_idx = np.random.randint(0, N, size=2)
        S2[x_idx, y_idx] *= -1
        dE = deltaE(S1, N, x_idx, y_idx)
        S1 = accept_reject_spin(S1, S2, dE, T)
        if i in save_iter:
            S_save.append(S1.copy())
            iterations_save.append(i)
    return S_save, iterations_save


@njit
def metropolis_hastings_Emag(
    S1: NDArray[np.int64],
    N: int,
    T: float,
) -> tuple[list[int], list[int]]:
    """Initialize random S and perform 1000 N² Metropolis Hastings steps."""
    iterations: int = 1000 * N**2
    mag_save: list = []
    E_save: list = []
    for i in range(iterations):
        S2: NDArray[np.float64] = S1.copy()
        x_idx, y_idx = np.random.randint(0, N, size=2)
        S2[x_idx, y_idx] *= -1
        dE = deltaE(S1, N, x_idx, y_idx)
        S1 = accept_reject_spin(S1, S2, dE, T)
        if i % 100 == 0:
            mag_save.append(magnetization(S1.copy()))
            E_save.append(ising_hamiltonian(S1.copy()))
    return mag_save, E_save


def magnetisation_temperature(
    S1: NDArray[np.int64],
    N: int,
    temps: list[float],
) -> list:
    """Calculate average magnetization as fucntion of temperature."""
    mag_avg: list = []
    for i in tqdm(range(len(temps))):
        S_init, _ = metropolis_hastings(S1, N, temps[i])
        S: NDArray[np.float64] = S_init[-1]
        mag_save, _ = metropolis_hastings_Emag(S, N, temps[i])
        mag_avg.append(np.mean(np.abs(mag_save)))
    return mag_avg


N: int = 50
N_mag: int = 10
S: NDArray[np.int64] = np.random.choice([-1, 1], size=(N, N))
S_mag: NDArray[np.int64] = np.random.choice([-1, 1], size=(N_mag, N_mag))
T: float = 0.5
S_save, iterations_save = metropolis_hastings(S, N, T)
T_array: NDArray[np.float64] = np.linspace(0.1, 5, 100)
mag_avg = magnetisation_temperature(S_mag, N_mag, T_array)
fig = plt.figure(figsize=(17, 11))
for i in range(len(S_save)):
    fig.add_subplot(2, 3, i + 1)
    plt.imshow(S_save[i], interpolation="nearest")
    plt.title(f"Spin configuration after {iterations_save[i]} iterations")
plt.tight_layout()
plt.show()
plt.figure(figsize=(6, 4))
plt.plot(T_array, mag_avg, ".")
plt.xlabel("Temperature")
plt.ylabel("Mean absolute magnetization.")
plt.show()
