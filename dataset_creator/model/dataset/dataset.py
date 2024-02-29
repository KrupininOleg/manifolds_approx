from typing import List
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree


@dataclass
class Dataset:
    values: npt.NDArray[np.float32]

    def __post_init__(self):
        self._kd_tree = KDTree(self.values)

    def get_closest_idx(self, point: npt.NDArray[np.float32], n: int) -> npt.NDArray[np.int64]:
        _, idx = self._kd_tree.query(point, k=n)
        return idx

    def get_closest_points(self, point: npt.NDArray[np.float32], n: int) -> npt.NDArray[np.float32]:
        idx = self.get_closest_idx(point, n)
        return self.values[idx]

    def get_idxs_in_radius(self, idx: int, r: float) -> npt.NDArray[np.int64]:
        point = self.values[idx]
        idx = self._kd_tree.query_ball_point(point, r)
        return np.array(idx, dtype=np.int64)
