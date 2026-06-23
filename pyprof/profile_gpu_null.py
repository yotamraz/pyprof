import warnings

from pyprof.profile_gpu_base import ProfileGPUBase


class ProfileGPUNull(ProfileGPUBase):
    def __init__(self, warn: bool = True):
        if warn:
            warnings.warn("No GPU backend available; GPU profiling disabled.", UserWarning, stacklevel=2)

    def __bool__(self) -> bool:
        return False

    def get_usage(self) -> tuple[float, float]:
        return 0.0, 0.0
