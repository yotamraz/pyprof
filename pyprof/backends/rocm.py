from pyprof.backends.base import GPUBackend


class ROCmBackend(GPUBackend):
    _driver = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            from pyrsmi import rocml
            rocml.smi_initialize()
            rocml.smi_shutdown()
            cls._driver = rocml
            return True
        except Exception:
            return False

    def __init__(self, device: int = 0):
        super().__init__(device)
        self._driver.smi_initialize()

    def get_usage(self) -> tuple:
        """Return (util_percent, mem_used_MB) for the configured ROCm device."""
        try:
            util_percent = self._driver.smi_get_device_utilization(self.device)
            mem_used_bytes = self._driver.smi_get_device_vram_usage(self.device)
        except AttributeError as exc:
            raise RuntimeError(
                f"ROCm SMI API does not expose expected methods "
                f"(smi_get_device_utilization / smi_get_device_vram_usage). "
                f"Check your pyrsmi version. Original error: {exc}"
            ) from exc
        mem_used_mb = mem_used_bytes / 1024 ** 2
        return float(util_percent), mem_used_mb
