"""Problem 2.3."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def piecewise(x: NDArray[np.float64], a: float) -> NDArray[np.float64]:
    """Return values of piecewise function f on grid x."""
    cond_list: list[NDArray[np.bool_]] = [x < 0, x > 0]
    c1, c2 = cond_list
    func_list: list[NDArray[np.float64]] = [
        np.exp(-x[c1]) + a * x[c1] - 1,
        x[c2] ** 2,
    ]
    return np.piecewise(x, cond_list, func_list)


def piecewise_deriv(x: NDArray[np.float64], a: float) -> NDArray[np.float64]:
    """Return values of derivative of f on grid x."""
    cond_list: list[NDArray[np.bool_]] = [x < 0, x > 0]
    c1, c2 = cond_list
    func_list: list[NDArray[np.float64]] = [-np.exp(-x[c1]) + a, 2 * x[c2]]
    return np.piecewise(x, cond_list, func_list)


def central_deriv(
    x: NDArray[np.float64],
    a: int,
    dx: float,
) -> NDArray[np.float64]:
    """Central derivative of f."""
    return (piecewise(x + dx, a) - piecewise(x - dx, a)) / (2 * dx)


def piecewise_deriv_robust(x: NDArray[np.float64], a: float, dx: float):
    """Numerical derivative of f that is robust around x=0.

    Ensures that point from x<0 are never used together with points x>0.
    """
    cond_list: list[NDArray[np.bool_]] = [x < 0, x > 0]
    c1, c2 = cond_list
    forward: NDArray[np.float64] = (piecewise(x + dx, a) - piecewise(x, a))[c2] / dx
    backward: NDArray[np.float64] = (piecewise(x, a) - piecewise(x - dx, a))[c1] / dx
    func_list: NDArray[np.float64] = [backward, forward]
    return np.piecewise(x, cond_list, func_list)


N = 1000
a_vals: NDArray[np.int64] = np.array([0, 1, 2])
x: NDArray[np.float64] = np.linspace(-1, 1, N)
dx: float = x[1] - x[0]

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    sharex=True,
    sharey=True,
    figsize=(12, 4),
)
for a in a_vals:
    ax1.plot(piecewise(x, a), label=f"a={a}")
    ax2.plot(piecewise_deriv(x, a), label=f"a={a}")
fig.supxlabel(r"$x$", fontsize=20)
ax1.set_ylabel(r"$f(x)$", fontsize=20)
ax2.set_ylabel(r"$f \ ^{\prime}(x)$", fontsize=20)
plt.legend()
plt.tight_layout()
plt.show()

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    sharex=True,
    figsize=(12, 4),
)

for a in a_vals:
    if a == 1:
        zorder: int = 3
    else:
        zorder: int = 1
    ax1.plot(
        x,
        np.abs(
            central_deriv(x, a, dx) - piecewise_deriv(x, a),
        ),
        label=f"a={a}",
        zorder=zorder,
        lw=3,
    )

    ax2.plot(
        x,
        np.abs(
            piecewise_deriv_robust(x, a, dx) - piecewise_deriv(x, a),
        ),
        label=f"a={a}",
        zorder=zorder,
        lw=3,
    )

fig.supxlabel(r"$x$", fontsize=20)
ax1.set_ylabel(
    r"$|f \ ^{\prime}_{central}-f \ ^{\prime}_{exact}|$",
    fontsize=15,
)
ax2.set_ylabel(
    r"$|f \ ^{\prime}_{robust}-f \ ^{\prime}_{exact}|$",
    fontsize=15,
)
plt.legend()
plt.tight_layout()
plt.show()
