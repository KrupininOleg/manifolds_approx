import numpy as np
import numpy.typing as npt
from scipy import interpolate


class Diode:
    def __init__(self, U: npt.NDArray[np.float32], I: npt.NDArray[np.float32]) -> None:
        self.i = interpolate.interp1d(U, I, "linear", fill_value="extrapolate")

    def r(self, u: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return u / self.i(u)


diode_1N4448 = Diode(
    U=np.array([-20.00, 0.0, 0.800, 0.864, 1.000, 1.200], dtype=np.float32),
    I=np.array([-25e-9, 0.0, 0.032, 0.050, 0.169, 0.352], dtype=np.float32),
)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    u = np.linspace(-1, 1.5, 1000)
    plt.plot(u, diode_1N4448.i(u))
    plt.plot([-20.00, 0.0, 0.800, 0.864, 1.000, 1.200], [-25e-6, 0.0, 0.032, 0.050, 0.169, 0.352])
    plt.plot(u, diode_1N4448.i(u))
    plt.show()
