"""Problem 1.2."""
from __future__ import annotations

from time import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import inv
from scipy.linalg import solve


def time_diff(
    func: Callable,
    args: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> tuple[NDArray[np.float64], float]:
    """Measures execution time of function call."""
    t1 = time()
    call = func(*args)
    t2 = time()
    return call, t2 - t1


def inv_solve(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve linear system by inverting the matrix."""
    inverse = inv(a)
    return inverse @ b


A1: NDArray[np.float64] = np.array(
    [
        [1, 3, -8],
        [3, 0, -1],
        [0, 10, 3],
    ],
)

b1: NDArray[np.float64] = np.array([1, 0, 7])
sol1, t1 = time_diff(solve, (A1, b1))
res1: NDArray[np.float64] = np.abs(A1 @ sol1 - b1)
print(
    f" \n a) \n Solution: {sol1} \n",
    f"Residual {res1} \n",
    f"Time taken: {t1} \n",
)

A2: NDArray[np.float64] = np.random.randn(1000, 1000)
b2: NDArray[np.float64] = np.random.randn(1000)
sol2, t2 = time_diff(solve, (A2, b2))
res2: NDArray[np.float64] = np.abs(A2 @ sol2 - b2)
print(
    f"b) \n Sum of Residuals {np.sum(res2)} \n",
    f"Time taken: {t2} \n",
)

sol3, t3 = time_diff(inv_solve, (A2, b2))
res3 = np.abs(A2 @ sol3 - b2)
print(
    f"c) \n Sum of Residuals {np.sum(res3)} \n",
    f"Time taken: {t3} \n",
)

A3: NDArray[np.float64] = np.random.randn(10, 10)
A3_inv, t_total_inv = time_diff(inv, (A3,))
t_total_sol: float = 0
print(time_diff(np.matmul, (A3, np.random.randn(10))))
for i in range(1000):
    b3: NDArray[np.float64] = np.random.randn(10)
    sol4, t4 = time_diff(solve, (A3, b3))
    sol5, t5 = time_diff(np.matmul, (A3_inv, b3))
    t_total_sol += t4
    t_total_inv += t5
print(f"d) \n Direct solve time: {t_total_sol}")
print(f"Inverse solve time: {t_total_inv}")
