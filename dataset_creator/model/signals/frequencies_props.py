from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.signal import spectrogram

from dataset_creator.model.signals.signals import Signal


@dataclass
class Spectrum:
    amplitudes: npt.NDArray[np.float32]
    frequencies: npt.NDArray[np.float32]

    @staticmethod
    def build(signal: Signal) -> "Spectrum":
        amplitudes = np.abs(np.fft.rfft(signal.values))
        frequencies = np.fft.rfftfreq(signal.values.size, signal.dt)
        return Spectrum(amplitudes, frequencies)


@dataclass
class Spectrogram:
    frequencies: npt.NDArray[np.float32]
    t: npt.NDArray[np.float32]
    amplitudes: npt.NDArray[np.float32]

    @staticmethod
    def build(signal: Signal) -> "Spectrogram":
        f, t, a = spectrogram(signal.values, 1 / signal.dt)
        return Spectrogram(f, t, a)
