"""Problem 1.1."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.special import lambertw


def f(z: NDArray[np.float64], x: float) -> NDArray[np.float64]:
    """Get function values for f.

    Where f is the function whose roots give the lambert W function.
    """
    return x * np.exp(x) - z


def central_diff(
    z: NDArray[np.float64],
    x: float,
    dx: float,
) -> NDArray[np.float64]:
    """Central derivative of f."""
    return (f(z, x + dx) - f(z, x - dx)) / 2 * dx


def my_lambertw(
    z: NDArray[np.float64],
    x_0: float,
    dx: float,
    max_iterations: int,
) -> float:
    """Return values of lambert W function.

    Uses Newton's root finding method.
    """
    x = x_0
    f_eval = f(z, x)
    n = 0
    df = central_diff(z, x, dx)
    while np.sum(np.abs(f_eval / df)) > 1e-5 and n < max_iterations:
        f_eval = f(z, x)
        df = central_diff(z, x, dx)
        x = x - f_eval / df
        n += 1
    return x


z: NDArray[np.float64] = np.arange(0, 150, 1)
x_0: float = 0.0
dx: float = 1

lambert_100_kwargs: dict[str, float | int | NDArray[np.float64]] = {
    "z": z,
    "x_0": x_0,
    "dx": dx,
    "max_iterations": 100,
}

lambert_nomax_kwargs: dict[str, float | int | NDArray[np.float64]] = {
    "z": z,
    "x_0": x_0,
    "dx": dx,
    "max_iterations": np.inf,
}

lambertw_100 = my_lambertw(**lambert_100_kwargs)
lambertw_nomax = my_lambertw(**lambert_nomax_kwargs)
lambertw_scipy = lambertw(z)

plt.plot(
    z,
    lambertw_100,
    label="Custom lambertW, 100 iterations",
    lw=3,
)
plt.plot(
    z,
    lambertw_scipy,
    label="Scipy lambertW implementation",
    lw=3,
    color="black",
)
plt.plot(
    z,
    lambertw_nomax,
    label="Custom lambertW, no max_iterations",
    linestyle="dashed",
)
plt.legend()
plt.show()
