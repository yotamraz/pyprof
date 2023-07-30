import os

import nvidia_smi

nvidia_smi.nvmlInit()


class ProfileGPU:
    def __init__(self, pid, device):
        assert device is not None, "device id should not be 'None'"
        self.device = device
        self.__pid = pid
        # init gpu handler
        self.gpu_handler = nvidia_smi.nvmlDeviceGetHandleByIndex(device)
        self.init_util = nvidia_smi.nvmlDeviceGetUtilizationRates(self.gpu_handler).gpu / 1.0
        self.init_memory_usage = nvidia_smi.nvmlDeviceGetMemoryInfo(self.gpu_handler).used / 1024**2

    def get_usage(self):
        """

        :return:
        """
        util = nvidia_smi.nvmlDeviceGetUtilizationRates(self.gpu_handler)
        memory_usage = nvidia_smi.nvmlDeviceGetMemoryInfo(self.gpu_handler)
        return util.gpu / 1.0, memory_usage.used / 1024**2
