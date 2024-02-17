from scipy.optimize import minimize
from dataset_creator.model.schemes.diode import diode_1N4448



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    def fun(u, E, R, Rc):
        i1, i2, i3 = diode_1N4448.i(u), diode_1N4448.i(-u), u / Rc
        i = i1 + i2 + i3
        return (u - E + i * R) ** 2

    R = 2200
    C = 10e-9
    w = 100
    Rc = 1 / (w * C)

    t = np.linspace(0, 1, 1000)
    f = np.sin(2 * np.pi * w * t, dtype=np.float32)

    f_fft = np.abs(np.fft.rfft(f))
    dt = t[1] - t[0]
    freq_axis = np.fft.rfftfreq(f.size, dt)
    # print(freq_axis[f_fft > 3 * f_fft.mean()])
    # plt.plot(freq_axis, f_fft)
    # plt.show()

    x = np.zeros_like(f)

    for i in range(f.size):
        x[i] = -minimize(fun, f[i], args=(f[i], R, Rc), method='BFGS').x[0]
    plt.plot(t, f)
    plt.plot(t, x)
    plt.show()