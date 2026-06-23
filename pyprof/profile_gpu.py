import warnings
import nvidia_smi

warnings.warn(
    "pyprof.profile_gpu is deprecated. Import ProfileGPUCUDA from pyprof.profile_gpu_cuda instead.",
    DeprecationWarning,
    stacklevel=2,
)

nvidia_smi.nvmlInit()

from pyprof.profile_gpu_cuda import ProfileGPUCUDA

ProfileGPU = ProfileGPUCUDA
