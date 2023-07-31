"""Problem 2.2."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve


def diff_coeffs(
    dx_coeffs: NDArray[np.int64],
    order: int,
) -> NDArray[np.float64]:
    """Get finite difference coefficients."""
    dim = len(dx_coeffs)
    A: NDArray[np.float64] = np.array([(dx_coeffs) ** n for n in range(dim)])
    b: NDArray[np.int64] = np.zeros(dim)
    b[order] = np.math.factorial(order)
    return solve(A, b)


coeff_dict: dict[str, NDArray[np.int64]] = {
    "First order": np.array([0, 1]),
    "Second order": np.array([-1, 1]),
    "Fourth order": np.array([-2, -1, 1, 2]),
    "Sixth order": np.array([-3, -2, -1, 1, 2, 3]),
}

print("\n a)")
deriv_order: int = 1
for key, value in coeff_dict.items():
    coeff = diff_coeffs(value, deriv_order)
    print(f"\n{key}:\n {coeff}:")

deriv_order2: int = 2
dx_coeffs2: NDArray[np.int64] = np.array([0, 1, 2, 3])
df1 = diff_coeffs(dx_coeffs2, deriv_order)
df2 = diff_coeffs(dx_coeffs2, deriv_order2)
print(f"\n b) \n First order derivative coefficients: {df1}")
print(f"\n Second order derivative coefficients: {df2}")
