from pyprof.backends.base import GPUBackend


class NVIDIABackend(GPUBackend):
    _driver = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            import nvidia_smi
            nvidia_smi.nvmlInit()
            nvidia_smi.nvmlShutdown()
            cls._driver = nvidia_smi
            return True
        except Exception:
            return False

    def __init__(self, device: int = 0):
        super().__init__(device)
        self._driver.nvmlInit()
        self._handle = self._driver.nvmlDeviceGetHandleByIndex(device)

    def get_usage(self) -> tuple[float, float]:
        """Return (util_percent, mem_used_MB) for the configured device."""
        util = self._driver.nvmlDeviceGetUtilizationRates(self._handle)
        memory_info = self._driver.nvmlDeviceGetMemoryInfo(self._handle)
        util_percent = util.gpu / 1.0
        mem_used_mb = memory_info.used / 1024 ** 2
        return util_percent, mem_used_mb
