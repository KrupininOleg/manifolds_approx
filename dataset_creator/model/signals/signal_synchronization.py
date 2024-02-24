import numpy as np
import numpy.typing as npt


def search_sync_signal(amplitudes):
    n_back = 3
    da = np.diff(amplitudes)
    dda = np.diff(da)
    k = 10
    max_dn = 3
    n = 50

    n_level = 0
    i0 = -1

    for i in range(1, da.size - 1):
        low_level = n_level % 2 == 1
        if low_level and da[i - 1] < da[i] > da[i + 1]:
            for j in range(i - 1, max(0, i - n_back - 1), -1):
                if dda[j - 1] < dda[j] > dda[j + 1]:
                    if abs((dda[j - 1] - dda[j]) / (dda[j - 2] - dda[j - 1])) > k:
                        real_idx = j + 2
                        if ((real_idx - i0) - n_level * n) < max_dn:
                            n_level += 1
                        else:
                            i0 = -1
                            n_level = 0
                        break
        elif not low_level and da[i - 1] > da[i] < da[i + 1]:
            for j in range(i, max(0, i - n_back), -1):
                if dda[j - 1] > dda[j] < dda[j + 1]:
                    if abs((dda[j - 1] - dda[j]) / (dda[j - 2] - dda[j - 1])) > k:
                        real_idx = j + 2
                        if n_level == 0:
                            i0 = real_idx
                            n_level += 1
                        else:
                            if ((real_idx - i0) - n_level * n) < max_dn:
                                n_level += 1
                            else:
                                i0 = -1
                                n_level = 0
                        break
        if n_level == 4:
            break

    return i0 + n_level * n
