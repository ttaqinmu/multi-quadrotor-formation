from typing import Callable

import numba as nb


def jitter(func: Callable, **kwargs):
    """Jits a function."""
    return nb.njit(func, **kwargs)
