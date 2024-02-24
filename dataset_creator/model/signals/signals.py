from typing import List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.io import wavfile
from scipy.interpolate import interp1d

from dataset_creator.model.signals.interval import Interval, stack_intervals
from dataset_creator.model.signals.noise import BaseNoise


class WaveFormType(int, Enum):
    sawtooth: int = 0
    sine: int = 1
    harmonic: int = 2


WaveFunType = Callable[[npt.NDArray[np.float32], npt.NDArray[np.float32]], npt.NDArray[np.float32]]


def sawtooth_wave(
    phase: npt.NDArray[np.float32],
    A: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    return A * ((2 * np.pi * phase % (2 * np.pi)) / (2 * np.pi) * 2 - 1)


def harmonic_wave(
    phase: npt.NDArray[np.float32],
    A: npt.NDArray[np.float32],
    harmonics: List[int],
) -> npt.NDArray[np.float32]:
    return np.sum([A * np.sin(h * 2 * np.pi * phase) for h in harmonics], axis=0)


def sine_wave(
    phase: npt.NDArray[np.float32],
    A: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    return harmonic_wave(phase, A, [1])


def get_wave_fun(wave: WaveFormType, **kwargs) -> WaveFunType:
    if wave is wave.sawtooth:
        return sawtooth_wave
    elif wave is wave.sine:
        return sine_wave
    elif wave is wave.harmonic:
        return partial(harmonic_wave, **kwargs)
    else:
        raise ValueError()


@dataclass
class Signal:
    # FIXME: dtypes
    values: npt.NDArray[np.float64]
    dt: float
    t0: float = 0.0
    name: str = "signal"

    @property
    def t(self) -> npt.NDArray[np.float64]:
        n = self.values.size
        return np.linspace(self.t0, self.t0 + (n - 1) * self.dt, n)

    @staticmethod
    def build_signal(
        wave_fun: WaveFunType,
        amplitudes: List[Interval],
        frequencies: List[Interval],
        k: float,
        dt: float,
        noise: Optional[BaseNoise] = None,
        t0: float = 0.0,
    ) -> "Signal":
        phase = stack_intervals(frequencies, dt)
        A = stack_intervals(amplitudes, dt)

        n = min(A.size, phase.size)
        phase, A = phase[:n], A[:n]

        values = wave_fun(phase, A)
        values *= k / values.max()

        if noise:
            values += noise.get_amplitudes(values.size, dt, k)

        return Signal(values, dt=dt, t0=t0)

    @staticmethod
    def load_from_wav(path: Path) -> "Signal":
        rate, values = wavfile.read(path)
        return Signal(values, 1 / rate, name=path.stem)

    def save_as_wav(self, path: Path) -> None:
        rate = round(1 / self.dt)
        wavfile.write(path, rate, self.values)

    @staticmethod
    def load_from_csv(path: Path, dt: Optional[float] = None) -> "Signal":
        # FIXME: dtypes
        data = np.genfromtxt(path, dtype=np.float64, delimiter=",")
        data = data[np.any(~np.isnan(data), axis=1)]
        t, a = data.T
        if dt is None:
            dt = (t[-1] - t[0]) / (t.size - 1)
        else:
            a_fun = interp1d(t, a, kind='linear', fill_value="extrapolate")
            n = round((t[-1] - t[0]) / dt) + 1
            t = np.linspace(t[0], t[0] + (n - 1) * dt, n)
            a = a_fun(t)

        return Signal(a, dt=dt, t0=t[0], name=path.stem)

    def save_as_csv(self, path: Path) -> None:
        data = np.vstack((self.t, self.values)).T
        n_decimals = max(4, int(abs(np.log10(self.dt))))
        np.savetxt(path, data, fmt=f"%.{n_decimals}f", delimiter=",")

    def cut(self, t0: Optional[float] = None, t1: Optional[float] = None) -> "Signal":
        i0 = round((t0 - self.t0) / self.dt) if t0 is not None else 0
        i1 = round((t1 - self.t0) / self.dt) if t1 is not None else self.values.size - 1
        return Signal(
            values=self.values[i0:i1 + 1].copy(),
            dt=self.dt,
            t0=t0,
        )
