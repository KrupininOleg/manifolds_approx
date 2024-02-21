from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dataset_creator.model.signals.signals import Signal, get_wave_fun, WaveFormType
import pyvista as pv


if __name__ == "__main__":
    u = Signal.load_from_csv(Path(r"C:\Users\Олег\Desktop\NonE\задачи от АА\data\orcad\raw\log2.csv"))
    x21 = Signal.load_from_csv(Path(r"C:\Users\Олег\Desktop\NonE\задачи от АА\data\orcad\scheme2\log2_scheme2_x1.csv"), dt=u.dt)
    x22 = Signal.load_from_csv(Path(r"C:\Users\Олег\Desktop\NonE\задачи от АА\data\orcad\scheme2\log2_scheme2_x2.csv"), dt=u.dt)

    points = np.vstack((x21.values, x22.values, u.values)).T
    plt = pv.Plotter()
    plt.add_axes()
    plt.add_points(points=points,  render_points_as_spheres=True, point_size=3.0)

    plt.show()
