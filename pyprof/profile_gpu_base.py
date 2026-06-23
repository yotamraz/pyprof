from abc import ABC, abstractmethod


class ProfileGPUBase(ABC):
    @abstractmethod
    def get_usage(self) -> tuple[float, float]:  # (utilization_%, memory_MB)
        raise NotImplementedError
