from pyprof.backends.base import GPUBackend


class AppleSiliconBackend(GPUBackend):
    _driver = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            import apple_smi
            # Probe the library to confirm it works on this machine.
            # Catch any exception — the import may succeed on non-Apple hardware
            # but the probe call will fail.
            apple_smi.get_gpu_utilization()
            cls._driver = apple_smi
            return True
        except Exception:
            return False

    def __init__(self, device: int = 0):
        super().__init__(device)

    def get_usage(self) -> tuple[float, float]:
        """Return (util_percent, mem_MB) for the Apple Silicon GPU."""
        try:
            util_percent = self._driver.get_gpu_utilization()
            mem_mb = self._driver.get_gpu_memory()
        except AttributeError as exc:
            raise RuntimeError(
                f"apple_smi does not expose the expected GPU API "
                f"(get_gpu_utilization / get_gpu_memory). "
                f"Check your apple_smi version. Original error: {exc}"
            ) from exc
        return float(util_percent), float(mem_mb)
