from dataclasses import dataclass
from abc import abstractmethod

import numpy as np
import numpy.typing as npt


@dataclass
class BaseNoise:
    @abstractmethod
    def get_amplitudes(self, n_samples: int, dt: float, base_amplitude: float) -> npt.NDArray[np.float32]:
        ...


@dataclass
class WhiteNoise(BaseNoise):
    level: float
    max_frequency: float

    def get_amplitudes(self, n_samples: int, dt: float, base_amplitude: float) -> npt.NDArray[np.float32]:
        _n_samples = n_samples if n_samples % 2 == 0 else n_samples + 1
        f_axis = np.fft.rfftfreq(_n_samples, dt)
        f = np.zeros(f_axis.size, dtype=np.complex64)
        [included_idx] = np.where(f_axis <= self.max_frequency)
        f[included_idx] = 1.0

        phases = 2 * np.pi * np.random.randn(f_axis.size)
        phases = np.cos(phases) + 1j * np.sin(phases)
        f *= phases
        noise = np.fft.irfft(f)

        noise_amplitude = base_amplitude * 10 ** (self.level / 20)
        noise *= noise_amplitude / np.abs(noise).max()

        return noise[:n_samples]
