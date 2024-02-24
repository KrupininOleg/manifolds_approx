from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition

from dataset_creator.model.signals.signals import Signal


if __name__ == "__main__":
    u = Signal.load_from_csv(Path(r"C:\Users\Олег\Desktop\NonE\задачи от АА\data\orcad\raw\log2.csv"))
    x1 = Signal.load_from_csv(Path(r"C:\Users\Олег\Desktop\NonE\задачи от АА\data\orcad\scheme1\log2_scheme1_x1.csv"), dt=u.dt)

    points = np.vstack((u.values, x1.values)).T
    shift = -np.mean(x1.values, axis=0)
    _points = points.copy()
    _points[:, 1] += shift

    pca = decomposition.PCA(n_components=1)
    pca.fit_transform(_points)

    v = pca.components_[0]
    k = v[1] / v[0]
    colors = np.abs(points[:, 1] - k * points[:, 0] + shift)
    colors = colors - colors.min()
    colors /= colors.max()

    plt.scatter(points[:, 0], points[:, 1], c=colors, s=3)
    plt.title("Схема1 log2")
    plt.xlabel("u (вход)")
    plt.ylabel("x1 (выход)")

    plt.show()

