import nvidia_smi

from pyprof.profile_gpu_base import ProfileGPUBase


class ProfileGPUCUDA(ProfileGPUBase):
    def __init__(self, device=0):
        assert device is not None, "device id should not be 'None'"
        self.device = device
        nvidia_smi.nvmlInit()
        self.gpu_handler = nvidia_smi.nvmlDeviceGetHandleByIndex(device)

    def get_usage(self) -> tuple[float, float]:
        util = nvidia_smi.nvmlDeviceGetUtilizationRates(self.gpu_handler)
        memory_usage = nvidia_smi.nvmlDeviceGetMemoryInfo(self.gpu_handler)
        return util.gpu / 1.0, memory_usage.used / 1024**2
