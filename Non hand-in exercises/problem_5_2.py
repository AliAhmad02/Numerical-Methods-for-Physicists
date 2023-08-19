"""Problem 5.2."""
from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def deriv_chem(
    x_0: NDArray[np.float64],
    k1: float,
    k2: float,
) -> NDArray[np.float64]:
    """Define derivative for chemical reaction."""
    dco2: float = -k1 * x_0[0] + k2 * x_0[1]
    dco3: float = k1 * x_0[0] - k2 * x_0[1]
    return np.array([dco2, dco3])


def explicit_euler(
    t_start: float,
    t_stop: float,
    dt: float,
    deriv: Callable,
    x_0: NDArray[np.float64],
    *args: tuple | None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Solve diff. eq using the explicit euler method."""
    N: int = int((t_stop - t_start) / dt + 1)
    t_vals: NDArray[np.float64] = np.linspace(t_start, t_stop, N)
    x: list[NDArray[np.float64]] = [x_0]
    for t in t_vals[:-1]:
        if args:
            x.append(x_0 + deriv(x_0, *args) * dt)
        else:
            x.append(x_0 + deriv(x_0) * dt)
        x_0 = x[-1]
    return t_vals, np.array(x)


def gillespie(
    N: int,
    t_start: int,
    t_end: int,
    k1: float,
    k2: float,
) -> NDArray[np.float64]:
    """Implement Gillespie algorithm for chemical reaction."""
    t = [t_start]
    N_CO2: list[int] = [N]
    N_H2CO3: list[int] = [0]
    while t[-1] < t_end:
        R: NDArray[np.float64] = np.array([k1 * N_CO2[-1], k2 * N_H2CO3[-1]])
        R_total: float = np.sum(R)
        if R_total == 0:
            break
        deltat: int = np.random.exponential(scale=1 / R_total)
        t.append(t[-1] + deltat)
        U: float = np.random.uniform(0, 1)
        if U * R_total <= R[0]:
            N_CO2.append(N_CO2[-1] - 1)
            N_H2CO3.append(N_H2CO3[-1] + 1)
        elif U * R_total > R[0]:
            N_CO2.append(N_CO2[-1] + 1)
            N_H2CO3.append(N_H2CO3[-1] - 1)
    return t, np.array(N_H2CO3) / N


t_start: int = 0
t_end: int = 20
dt: float = 0.02
x_0: NDArray[np.float64] = np.array([1, 0])
k1: float = 1e-3
k2: float = 1.0
t, sol = explicit_euler(t_start, t_end, dt, deriv_chem, x_0, k1, k2)
diff1: NDArray[np.float64] = -k1 * sol[:, 0] + k2 * sol[:, 1]
diff2: NDArray[np.float64] = k1 * sol[:, 0] - k2 * sol[:, 1]
N1: int = 1000
N2: int = 10_000
N3: int = 100_000
N4: int = 1_000_000
N_array: NDArray[np.float64] = np.array([N1, N2, N3, N4])

plt.figure(figsize=(6, 4))
plt.plot(t, diff1, label=r"$c^{\prime}_{CO_2}$")
plt.plot(t, diff2, label=r"$c^{\prime}_{H_2 CO_3}$")
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.legend()
plt.tight_layout()
plt.show()

for N in N_array:
    fig, ax = plt.subplots(3, 2, figsize=(18, 8))
    for ax in ax.flatten()[:-1]:
        ax.step(t, sol[:, 1], label=r"$c_{H_2 O_3}$", where="post")
        ax.step(
            *gillespie(N, t_start, t_end, k1, k2),
            where="post",
            label=r"$c_{H_2 O_3} (Gillespie)$",
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Concentration")
        ax.legend()
    fig.suptitle(f"Gillespie for N={N}")
    plt.tight_layout()
    plt.show()
