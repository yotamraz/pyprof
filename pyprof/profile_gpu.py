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
        self.init_util = nvidia_smi.nvmlDeviceGetUtilizationRates(self.gpu_handler).gpu / 1.0
        self.init_memory_usage = nvidia_smi.nvmlDeviceGetMemoryInfo(self.gpu_handler).used / 1024**2
