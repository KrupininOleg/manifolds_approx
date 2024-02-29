import numpy as np
import numpy.typing as npt
from numba import njit


@njit("(f4[:, :])(f4[:, :], i8, f4)", cache=True)
def get_tangent(values: npt.NDArray[np.float32], idx: int, length: float) -> npt.NDArray[np.float32]:
    v = np.zeros((2, values.shape[1]), dtype=np.float32)
    n = 0

    if idx < len(values) - 1:
        v[1] += values[idx + 1] - values[idx]
        n += 1
    if idx > 0:
        v[1] += values[idx] - values[idx - 1]
        n += 1

    v[0] = values[idx]
    v[1] /= np.float32(n)
    v[1] = v[1] / (np.linalg.norm(v[1]) / length) + v[0]

    return v


@njit("(f4[:, :, :])(f4[:, :], i8[:], f4)")
def get_tangents_on_idxs(values: npt.NDArray[np.float32], idxs: npt.NDArray[np.int64], length: float) -> npt.NDArray[np.float32]:
    tangents = np.empty((len(idxs), 2, values.shape[1]), dtype=np.float32)

    for i in range(len(idxs)):
        tangents[i] = get_tangent(values, idxs[i], length)

    return tangents


@njit("(f4[:, :, :])(f4[:, :], i8, i8, f4)")
def get_tangents_on_interval(values: npt.NDArray[np.float32], start_idx: int, end_idx: int, length: float) -> npt.NDArray[np.float32]:
    n = end_idx - start_idx + 1
    tangents = np.empty((n, 2, values.shape[1]), dtype=np.float32)

    for i in range(n):
        tangents[i] = get_tangent(values, i + start_idx, length)

    return tangents
