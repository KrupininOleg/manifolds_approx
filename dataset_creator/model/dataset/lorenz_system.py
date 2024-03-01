from enum import Enum
from typing import List

import numpy as np
import numpy.typing as npt
from numba import njit

from dataset_creator.model.dataset.dataset import Dataset


class LorenzSystemVariables(int, Enum):
    t: int = 0
    x1: int = 1
    x2: int = 2
    x3: int = 3


@njit("(f4[:, :])(f4, f4, f4, f4, f4, f4, f4, f4, f4)", cache=True)
def lorenz_system(
    x1_0: float,
    x2_0: float,
    x3_0: float,
    t: float,
    dt: float,
    sigma: float,
    rho: float,
    beta: float,
    mu: float,
) -> npt.NDArray[np.float32]:
    n = int(t / dt)
    points = np.zeros((n, 4), dtype=np.float32)
    points[0, 1:] = x1_0, x2_0, x3_0

    for i in range(1, n):
        t0, x1_0, x2_0, x3_0 = points[i - 1]

        points[i, 0] = t0 + dt
        points[i, 1] = x1_0 + (sigma * (x2_0 - x1_0)) * dt
        points[i, 2] = x2_0 + (rho * x1_0 - x2_0 - mu * x1_0 * x3_0) * dt
        points[i, 3] = x3_0 + (mu * x1_0 * x2_0 - beta * x3_0) * dt

    return points


def get_dataset_by_lorenz_system(
    include_variables: List[LorenzSystemVariables],
    x1_0: float,
    x2_0: float,
    x3_0: float,
    t: float,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    mu: float = 1.0,
) -> Dataset:
    points = lorenz_system(x1_0, x2_0, x3_0, t, dt, sigma, rho, beta, mu)
    include_variables_idxs = [v.value for v in include_variables]
    return Dataset(points[:, include_variables_idxs])
