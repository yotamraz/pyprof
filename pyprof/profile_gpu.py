from __future__ import annotations

from importlib import import_module
from typing import Any, Optional


class ProfileGPU:
    """
    GPU usage profiler.

    Currently supports NVIDIA GPUs through NVML (``nvidia-ml-py3``).
    On unsupported systems (for example Apple Silicon or AMD-only hosts),
    initialization raises ``RuntimeError`` with a clear explanation.
    """

    def __init__(self, pid: int, device: int) -> None:
        if device is None:
            raise ValueError("device id should not be None")

        self.device = device
        self._pid = pid
        self._nvidia_smi = self._load_nvidia_smi()
        self._nvidia_smi.nvmlInit()
        self._gpu_handler = self._nvidia_smi.nvmlDeviceGetHandleByIndex(device)

    @property
    def backend_name(self) -> str:
        return "nvidia-nvml"

    @classmethod
    def get_unavailable_reason(cls) -> Optional[str]:
        """
        Return ``None`` when NVIDIA NVML is available, otherwise a reason string.
        """
        try:
            nvidia_smi = cls._load_nvidia_smi()
        except RuntimeError as error:
            return str(error)

        try:
            nvidia_smi.nvmlInit()
            _ = nvidia_smi.nvmlDeviceGetCount()
        except Exception as error:  # pragma: no cover - vendor-specific errors
            return f"nvidia nvml unavailable: {error}"
        finally:
            try:
                nvidia_smi.nvmlShutdown()
            except Exception:  # pragma: no cover - best effort cleanup
                pass

        return None

    @staticmethod
    def _load_nvidia_smi() -> Any:
        try:
            return import_module("nvidia_smi")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "nvidia-ml-py3 is not installed; GPU profiling is unavailable on this host"
            ) from error

    def get_usage(self) -> tuple[float, float]:
        util = self._nvidia_smi.nvmlDeviceGetUtilizationRates(self._gpu_handler)
        memory_usage = self._nvidia_smi.nvmlDeviceGetMemoryInfo(self._gpu_handler)
        return float(util.gpu), float(memory_usage.used / 1024**2)
