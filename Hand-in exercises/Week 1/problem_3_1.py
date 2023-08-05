"""Problem 3.1."""
from __future__ import annotations

from typing import Callable

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


def explicit_euler(
    t_start: float,
    t_stop: float,
    dt: float,
    deriv: Callable,
    *args: tuple,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Solve diff. eq using the explicit euler method."""
    N: int = int((t_stop - t_start) / dt + 1)
    t_vals: NDArray[np.float64] = np.linspace(t_start, t_stop, N)
    x_0: float = 0
    x: list[float] = [x_0]
    for t in t_vals[:-1]:
        x.append(x_0 + deriv(x_0, t, *args) * dt)
        x_0 = x[-1]
    return t_vals, np.array(x)


def implicit_euler(
    t_start: float,
    t_stop: float,
    dt: float,
    alpha: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Solve diff. eq using the implicit euler method."""
    N: int = int((t_stop - t_start) / dt + 1)
    t_vals: NDArray[np.float64] = np.linspace(t_start, t_stop, N)
    x_0: float = 0
    x: list[float] = [x_0]
    for t in t_vals[:-1]:
        x.append(
            1 / (1 + alpha * dt) * (x_0 + alpha * dt * np.sin(t + dt)),
        )
        x_0 = x[-1]
    return t_vals, np.array(x)


def plotting(
    t: NDArray[np.float64],
    x: NDArray[np.float64],
    euler_ex: NDArray[np.float64],
    euler_imp: NDArray[np.float64],
    sharex: bool,
    sharey: bool,
) -> None:
    """Plot exact solution and numerical solutions."""
    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(12, 12),
        sharex=sharex,
        sharey=sharey,
    )
    ax1.plot(t, euler_ex, label="Explicit Euler", lw=2, color="orange")
    ax2.plot(t, x, label="Exact", lw=2, color="green")
    ax3.plot(t, euler_imp, label="Implicit euler", lw=2, color="blue")
    ax1.legend(fontsize=15)
    ax2.legend(fontsize=15)
    ax3.legend(fontsize=15)
    fig.supxlabel(r"$t$", fontsize=25)
    fig.supylabel(r"$x(t)$", fontsize=25)
    plt.tight_layout()
    plt.show()


t_start: int = 0
t_stop: int = 100
dt: float = 0.01
alpha: float = 0.1
alpha1: int = 200.1

t, x_euler_ex = explicit_euler(t_start, t_stop, dt, dxdt, alpha)
_, x_euler_imp = implicit_euler(t_start, t_stop, dt, alpha)
x_exact = x_analytic(t, alpha)

_, x_euler_ex1 = explicit_euler(t_start, t_stop, dt, dxdt, alpha1)
_, x_euler_imp1 = implicit_euler(t_start, t_stop, dt, alpha1)
x_exact1 = x_analytic(t, alpha1)

plotting(t, x_exact, x_euler_ex, x_euler_imp, True, True)
plotting(t, x_exact1, x_euler_ex1, x_euler_imp1, True, False)
