from pyprof.backends.base import GPUBackend
from pyprof.backends.nvidia import NVIDIABackend
from pyprof.backends.rocm import ROCmBackend
from pyprof.backends.apple_silicon import AppleSiliconBackend

_BACKEND_PRIORITY = [NVIDIABackend, ROCmBackend, AppleSiliconBackend]


def detect_backend(device: int = 0):
    """Probe each GPU backend in priority order and return the first available one.

    Returns an instantiated GPUBackend, or None if no supported GPU is found.
    """
    for cls in _BACKEND_PRIORITY:
        if cls.is_available():
            return cls(device=device)
    return None


__all__ = [
    "GPUBackend",
    "NVIDIABackend",
    "ROCmBackend",
    "AppleSiliconBackend",
    "detect_backend",
]
