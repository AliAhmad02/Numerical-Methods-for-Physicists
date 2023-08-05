"""Problem 3.3."""
from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from problem_3_1 import explicit_euler


def diff(f: float, t: float):
    """Return derivative of f."""
    return 1 + np.sin(t) * f


def RK4(
    t_start: float,
    t_stop: float,
    dt: float,
    dfdt: Callable,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Solve diff eq. using 4th order Runge Kutta method."""
    N: int = int((t_stop - t_start) / dt + 1)
    t_vals: NDArray[np.float64] = np.linspace(t_start, t_stop, N)
    f0: float = 0
    f: list[float] = [f0]
    for t in t_vals[:-1]:
        k1: float = dfdt(f0, t)
        k2: float = dfdt(f0 + 1 / 2 * k1 * dt, t + 1 / 2 * dt)
        k3: float = dfdt(f0 + 1 / 2 * k2 * dt, t + 1 / 2 * dt)
        k4: float = dfdt(f0 + k3 * dt, t + dt)
        f.append(f0 + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * dt)
        f0 = f[-1]
    return t_vals, np.array(f)


t_start: int = 0
t_stop: int = 15
dt: float = 0.001
dt1: int = 1
dt2: float = 0.25

RK_sol = RK4(t_start, t_stop, dt, diff)
Euler_sol = explicit_euler(t_start, t_stop, dt, diff)
RK_sol1 = RK4(t_start, t_stop, dt1, diff)
Euler_sol1 = explicit_euler(t_start, t_stop, dt2, diff)

plt.figure(figsize=(6, 4))
plt.plot(*Euler_sol, label="Explicit euler solution", lw=3)
plt.plot(*RK_sol, "--", label="4th order Runge-Kutta solution", lw=3)
plt.xlabel("t", fontsize=15)
plt.ylabel(r"$f^{ \ \prime}(x)$", fontsize=15)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(*RK_sol, label=r"RK $\Delta t = 0.001$", lw=3)
plt.plot(*RK_sol1, "--", label=r"RK $\Delta t = 1$", lw=3)
plt.xlabel("t", fontsize=15)
plt.ylabel(r"$f^{ \ \prime}(x)$", fontsize=15)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(*Euler_sol, label=r"Euler $\Delta t = 0.001$", lw=3)
plt.plot(*Euler_sol1, "--", label=r"Euler $\Delta t = 0.25$", lw=3)
plt.xlabel("t", fontsize=15)
plt.ylabel(r"$f^{ \ \prime}(x)$", fontsize=15)
plt.legend()
plt.tight_layout()
plt.show()
