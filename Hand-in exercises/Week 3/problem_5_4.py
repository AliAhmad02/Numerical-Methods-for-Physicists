"""Problem 5.4."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve
from scipy.ndimage import generate_binary_structure
from tqdm import tqdm


def ising_hamiltonian(S: NDArray[np.float64]) -> float:
    """Take an NxN spin matrix S and return its Hamiltonian."""
    # Generates a 3x3 boolean matrix with False on the 4 corners
    kernel: NDArray[np.bool_] = generate_binary_structure(2, 1)
    # Set the middle value in the kernel to False.
    kernel[1, 1] = False
    # We now have a kernel where only the 4 neighbors are True
    # We convolve this with our lattice. 'wrap' accounts for periodicity.
    H: NDArray[np.float64] = -S * convolve(S, kernel, mode="wrap")
    # We sum and divide by two to account for double-counting.
    return H.sum() / 2


def magnetization(S: NDArray[np.float64]) -> float:
    """Calculate the magnetization of a spin configuration S."""
    return np.mean(S)


def deltaE(S1: NDArray[np.float64], S2: NDArray[np.float64]) -> float:
    """Calculate difference of energy of two spin configurations."""
    return ising_hamiltonian(S2) - ising_hamiltonian(S1)


def accept_reject_spin(
    S1: NDArray[np.float64],
    S2: NDArray[np.float64],
    T: float,
) -> NDArray[np.float64]:
    """Accept or reject a spin flip based on change in energy."""
    energy_change = deltaE(S1, S2)
    alpha = min(1, np.exp(-energy_change / T))
    U = np.random.uniform(0, 1)
    if U <= alpha:
        S = S2
    else:
        S = S1
    return S


def metropolis_hastings(
    S1: NDArray[np.int64],
    N: int,
    T: float,
) -> NDArray[np.float64]:
    """Initialize random S and perform 1000 N² Metropolis Hastings steps."""
    iterations: int = 1000 * N**2
    iterations_save: list = []
    bruh = [
        1,
        int((1 / 100) * iterations),
        int((1 / 50) * iterations),
        int((1 / 20) * iterations),
        int((1 / 5) * iterations),
        int(iterations),
    ]
    S_save = []
    for i in tqdm(range(iterations)):
        S2: NDArray[np.float64] = S1.copy()
        x_idx, y_idx = np.random.randint(0, N, size=2)
        S2[x_idx, y_idx] *= -1
        S1 = accept_reject_spin(S1, S2, T)
        if i in bruh:
            S_save.append(S1.copy())
            iterations_save.append(i)
    return np.array(S_save), np.array(iterations_save)


N: int = 20
S: NDArray[np.int64] = np.random.choice([-1, 1], size=(N, N))
T: float = 0.5
S_save, iterations_save = metropolis_hastings(S, N, T)
for i in range(len(S_save)):
    plt.imshow(S_save[i], interpolation="nearest")
    plt.title(f"Spin configuration after {iterations_save[i]} iterations")
    plt.show()
