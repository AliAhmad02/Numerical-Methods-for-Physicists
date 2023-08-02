"""Problem 3.1."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def dxdt(
    x: NDArray[np.float64],
    t: float,
    alpha: float,
) -> NDArray[np.float64]:
    """Return values of dxdt."""
    return alpha * (np.sin(t) - x)


def x_analytic(t: NDArray[np.float64], alpha: float) -> NDArray[np.float64]:
    """Return values for analytic solution of diff eq."""
    prefactor: float = alpha / (1 + alpha**2)
    return prefactor * (np.exp(-alpha * t) - np.cos(t) + alpha * np.sin(t))


def explicit_euler(t_start: float, t_stop: float, dt: float, alpha: float):
    """Solve diff. eq using the explicit euler method."""
    N: int = int((t_stop - t_start) / dt + 1)
    t_vals: NDArray[np.float64] = np.linspace(t_start, t_stop, N)
    x_0: float = 0
    x: list[float] = [x_0]
    for t in t_vals[:-1]:
        x.append(x_0 + dxdt(x_0, t, alpha) * dt)
        x_0 = x[-1]
    return t_vals, np.array(x)


t_start: int = 0
t_stop: int = 100
dt: float = 0.01
alpha: float = 0.1
t, x_euler_ex = explicit_euler(t_start, t_stop, dt, alpha)
x_exact = x_analytic(t, alpha)

plt.plot(t, x_euler_ex, label="Euler", lw=3)
plt.plot(t, x_exact, label="Exact", lw=2, ls="--", dashes=(4, 3))
plt.legend()
plt.show()
