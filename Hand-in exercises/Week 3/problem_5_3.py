"""Problem 5.3."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def gillespie_DR(
    t_start: int,
    time_steps: int,
    R0: int,
    D0: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Implement Gillespie method for Democrat and Republican voters."""
    t = [t_start]
    R: list[int] = [R0]
    D: list[int] = [D0]
    while len(t) < time_steps:
        DR_rand: float = 0.1 * D[-1]
        RD_rand: float = 0.1 * R[-1]
        DR_con: float = 0.01 * D[-1] * R[-1]
        RD_con: float = 0.01 * R[-1] * D[-1]
        r: NDArray[np.float64] = np.array([DR_rand, RD_rand, DR_con, RD_con])
        r_total: float = np.sum(r)
        deltat: int = np.random.exponential(scale=1 / r_total)
        t.append(t[-1] + deltat)
        U: float = np.random.uniform(0, r_total)
        event_idx: int = np.argmax(np.cumsum(r) >= U)
        if event_idx % 2 == 0:
            D.append(D[-1] - 1)
            R.append(R[-1] + 1)
        else:
            R.append(R[-1] - 1)
            D.append(D[-1] + 1)
    return np.array(t), np.array(R), np.array(D)


def gillespie_DUR(
    t_start: int,
    time_steps: int,
    R0: int,
    D0: int,
    U0: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Implement Gillespie method for D, R and U voters."""
    t = [t_start]
    R: list[int] = [R0]
    D: list[int] = [D0]
    U: list[int] = [U0]
    while len(t) < time_steps:
        DU_rand: float = 0.1 * D[-1]
        RU_rand: float = 0.1 * R[-1]
        UD_rand: float = 0.05 * U[-1]
        UR_rand: float = 0.05 * U[-1]
        DU_con: float = 0.01 * D[-1] * R[-1]
        UR_con: float = 0.01 * U[-1] * R[-1]
        RU_con: float = 0.01 * R[-1] * D[-1]
        UD_con: float = 0.01 * U[-1] * D[-1]
        r: NDArray[np.float64] = np.array(
            [
                DU_rand,
                RU_rand,
                UD_rand,
                UR_rand,
                DU_con,
                UR_con,
                RU_con,
                UD_con,
            ],
        )
        r_total: float = np.sum(r)
        deltat: int = np.random.exponential(scale=1 / r_total)
        t.append(t[-1] + deltat)
        u: float = np.random.uniform(0, r_total)
        event_idx: int = np.argmax(np.cumsum(r) >= u)
        if event_idx == 0 or event_idx == 4:
            D.append(D[-1] - 1)
            U.append(U[-1] + 1)
            R.append(R[-1])
        elif event_idx == 1 or event_idx == 6:
            R.append(R[-1] - 1)
            U.append(U[-1] + 1)
            D.append(D[-1])
        elif event_idx == 2 or event_idx == 7:
            U.append(U[-1] - 1)
            D.append(D[-1] + 1)
            R.append(R[-1])
        elif event_idx == 3 or event_idx == 5:
            U.append(U[-1] - 1)
            R.append(R[-1] + 1)
            D.append(D[-1])
    return np.array(t), np.array(R), np.array(D)


t_start: int = 0
time_steps = 500_000
R0: int = 25
D0: int = 25

R0_un: int = 0
U0: int = 50
D0_un: int = 0

t1, R1, D1 = gillespie_DR(t_start, time_steps, R0, D0)
plt.figure(figsize=(6, 4))
plt.step(t1, R1, label="R(t)", where="post")
plt.step(t1, D1, label="D(t)", where="post")
plt.xlabel("Time", fontsize=15)
plt.ylabel("# of voters", fontsize=15)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

t2, R2, D2 = gillespie_DUR(
    t_start,
    time_steps,
    R0_un,
    D0_un,
    U0,
)
plt.figure(figsize=(6, 4))
plt.step(t2, R2, label="R(t)", where="post")
plt.step(t2, D2, label="D(t)", where="post")
plt.xlabel("Time", fontsize=15)
plt.ylabel("# of voters", fontsize=15)
plt.legend(loc="center left")
plt.tight_layout()
plt.show()
