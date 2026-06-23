import warnings
import nvidia_smi

warnings.warn(
    "pyprof.profile_gpu is deprecated. Import ProfileGPUCUDA from pyprof.profile_gpu_cuda instead.",
    DeprecationWarning,
    stacklevel=2,
)

nvidia_smi.nvmlInit()

from pyprof.profile_gpu_cuda import ProfileGPUCUDA


class ProfileGPU(ProfileGPUCUDA):
    """Backward-compat wrapper preserving the original (pid, device) constructor signature."""

    def __init__(self, pid=None, device=0):
        super().__init__(device=device)
