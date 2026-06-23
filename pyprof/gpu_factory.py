import importlib.util
import shutil
import sys


def create_gpu_profiler(backend: str = 'auto', device: int = 0):
    """
    Create and return a GPU profiler backend.

    Parameters
    ----------
    backend : str
        'auto', 'cuda', 'rocm', 'mps', or 'none'
    device : int
        GPU device index (default 0)

    Returns
    -------
    ProfileGPUBase subclass instance. Returns ProfileGPUNull (falsy) when no
    GPU is available or backend='none'. Never returns None.
    """
    from pyprof.profile_gpu_null import ProfileGPUNull

    if backend == 'none':
        return ProfileGPUNull(warn=False)

    cuda_available = importlib.util.find_spec('nvidia_smi') is not None
    if backend == 'cuda' or (backend == 'auto' and cuda_available):
        if backend == 'cuda' and not cuda_available:
            raise RuntimeError("Backend 'cuda' requested but nvidia-ml-py3 is not installed.")
        from pyprof.profile_gpu_cuda import ProfileGPUCUDA
        return ProfileGPUCUDA(device=device)

    rocm_available = importlib.util.find_spec('amdsmi') is not None
    if backend == 'rocm' or (backend == 'auto' and rocm_available):
        if backend == 'rocm' and not rocm_available:
            raise RuntimeError("Backend 'rocm' requested but amdsmi is not installed.")
        from pyprof.profile_gpu_rocm import ProfileGPUROCm
        return ProfileGPUROCm(device=device)

    mps_available = bool(sys.platform == 'darwin' and shutil.which('powermetrics'))
    if backend == 'mps' or (backend == 'auto' and mps_available):
        if backend == 'mps' and not mps_available:
            raise RuntimeError("Backend 'mps' requested but not on macOS or powermetrics not found.")
        from pyprof.profile_gpu_mps import ProfileGPUMPS
        return ProfileGPUMPS(device=device)

    if backend == 'auto':
        return ProfileGPUNull(warn=True)

    raise RuntimeError(
        f"Unknown backend: '{backend}'. Valid values: 'auto', 'cuda', 'rocm', 'mps', 'none'."
    )
