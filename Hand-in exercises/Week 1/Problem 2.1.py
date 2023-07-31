"""Problem 2.1."""
from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def forward_diff(
    func: Callable,
    x: NDArray[np.float64],
    dx: float,
) -> NDArray[np.float64]:
    """Differentiate function using forward difference method on grid x."""
    return (func(x + dx) - func(x)) / dx


def generalized_diff(
    func: Callable,
    x: float,
    dx: float,
    coeff: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate the numerical derivative using the coefficients."""
    fs = np.array([func(x + a * dx) for a in range(-3, 4)])
    return 1 / dx * np.dot(coeff, fs)


N: int = 20
x: NDArray[np.float64] = np.linspace(0, 2 * np.pi, N, endpoint=False)
dx: float = 2 * np.pi / N
diffx = forward_diff(np.sin, x, dx)
cosx: NDArray[np.float64] = np.cos(x)

plt.figure(figsize=(5, 4))
plt.plot(x, diffx, label="Numerical differentiation")
plt.plot(x, cosx, label="Exact result")
plt.ylabel(r"$f'(x)$", fontsize=15)
plt.xlabel(r"$x$", fontsize=15)
plt.title("First order forward difference method vs exact result")
plt.legend()
plt.show()

coeff_mat = [
    np.array([0, 0, 0, -1, 1, 0, 0]),
    np.array([0, 0, -0.5, 0, 0.5, 0, 0]),
    np.array([0, 1 / 12, -2 / 3, 0, 2 / 3, -1 / 12, 0]),
    np.array([-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60]),
]
max_abs_err: list[float] = []
plt.figure(figsize=(5, 4))
plt.plot(x, cosx, label="Exact result")
for index, array in enumerate(coeff_mat):
    deriv = generalized_diff(np.sin, x, dx, array)
    max_abs_err.append(
        np.amax(
            np.abs(deriv - cosx),
        ),
    )
    plt.plot(x, deriv, label=rf"$O(x^{2*index})$")
plt.ylabel(r"$f'(x)$", fontsize=15)
plt.xlabel(r"$x$", fontsize=15)
plt.title("Different numerical differentiation methods vs exact result")
plt.legend()
plt.show()
print(f"\n Maximum absolute errors: \n {max_abs_err}")

Nlog: NDArray[np.int64] = np.logspace(1, 6, 50, dtype=int)
max_abs_err_N: list[float] = []

for index, array in enumerate(coeff_mat):
    for N1 in Nlog:
        x1: NDArray[np.float64] = np.linspace(0, 2 * np.pi, N1, endpoint=False)
        dx1: float = 2 * np.pi / N1
        deriv = generalized_diff(np.sin, x1, dx1, array)
        cosx1: NDArray[np.float64] = np.cos(x1)
        max_abs_err_N.append(
            np.amax(
                np.abs(deriv - cosx1),
            ),
        )

max_abs_err_N: NDArray[np.float64] = np.array(max_abs_err_N).reshape(-1, 50)

plt.figure(figsize=(6, 4))
for i in range(len(max_abs_err_N)):
    plt.plot(Nlog, max_abs_err_N[i], label=rf"$O(x^{2*i})$")
plt.axhline(
    np.amin(max_abs_err_N[1]),
    linestyle="dashed",
    label=r"$O(x²)$ best accuracy",
    color="black",
)
plt.text(
    7,
    np.amin(max_abs_err_N[1]) + 1e-11,
    r"$1.7 \times 10^{-11}$",
    color="black",
    fontsize=13,
)
plt.axhline(
    np.amin(max_abs_err_N[-1]),
    linestyle="dashed",
    label=r"$O(x⁶)$ best accuracy",
    color="red",
)
plt.text(
    7,
    np.amin(max_abs_err_N[-1]) + 5e-14,
    r"$4.6 \times 10^{-14}$",
    color="red",
    fontsize=13,
)
plt.xscale("log")
plt.yscale("log")
plt.title("Absolute error as a function of N (log-log scale)")
plt.xlabel("N", fontsize=15)
plt.ylabel("Absolute error", fontsize=15)
plt.legend()
plt.show()
