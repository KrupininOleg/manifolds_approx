import numpy as np
import numpy.typing as npt
from sklearn import decomposition


def get_linear_deviation(data: npt.NDArray[np.float32], norm: bool = True) -> npt.NDArray[np.float32]:
    data_shifted = data - np.mean(data)

    pca = decomposition.PCA(n_components=1)
    pca.fit_transform(data_shifted)

    v = pca.components_[0]

    if data.shape[1] == 2:
        k0 = v[1] / v[0]
        line = k0 * data_shifted[:, 0]
        dists = np.abs(data_shifted[:, 1] - line)
    elif data.shape[1] == 3:
        k0 = v[2] / v[0]
        k1 = v[2] / v[1]
        line = k0 * data_shifted[:, 0] + k1 * data_shifted[:, 1]
        dists = np.abs(data_shifted[:, 1] - line)
    else:
        raise ValueError

    if norm:
        dists -= dists.min()
        dists /= dists.max()

    return dists