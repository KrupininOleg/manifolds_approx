from typing import Optional, List
from enum import Enum
from dataclasses import dataclass
from abc import abstractmethod

import numpy as np
import numpy.typing as npt


class ModulationType(int, Enum):
    linear: int = 0
    logarithmic: int = 1


@dataclass
class RangeBase:
    start: float
    end: Optional[float] = None
    modulation: ModulationType = ModulationType.linear

    @abstractmethod
    def fill(self, duration: float, dt: float) -> npt.NDArray[np.float32]:
        ...


class RangeFrequencies(RangeBase):
    def fill(self, duration: float, dt: float) -> npt.NDArray[np.float32]:
        t = np.linspace(0, duration, int(duration / dt) + 1)
        if self.start == self.end or self.end is None:
            values = self.start * t
        elif self.modulation is ModulationType.linear:
            T = t[-1] - t[0]
            c = (self.end - self.start) / T
            values = c / 2 * t ** 2 + self.start * t
        elif self.modulation is ModulationType.logarithmic:
            T = t[-1] - t[0]
            k = self.end / self.start
            c = T / np.log(k)
            values = c * self.start * k ** (t / T) - 1.0
        else:
            raise ValueError()

        return values


class RangeAmplitudes(RangeBase):
    def fill(self, duration: float, dt: float) -> npt.NDArray[np.float32]:
        n = int(duration / dt) + 1
        if self.start == self.end or self.end is None:
            values = np.repeat(self.start, n)
        elif self.modulation is ModulationType.linear:
            values = np.linspace(self.start, self.end, n)
        elif self.modulation is ModulationType.logarithmic:
            t = np.arange(1, n + 1)
            s, e = self.start, self.end
            if is_decrease := s > e:
                s, e = e, s
            values = s + (e - s) / np.log(n) * np.log(t)
            if is_decrease:
                values = values[::-1]
        else:
            raise ValueError()

        return values


@dataclass
class Interval:
    duration: float
    values: RangeBase


def stack_intervals(intervals: List[Interval], dt: float) -> npt.NDArray[np.float32]:
    values = []
    for i, v in enumerate(intervals):
        values.append(v.values.fill(v.duration, dt))
    return np.hstack(values)
