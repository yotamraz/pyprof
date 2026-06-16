from abc import ABC, abstractmethod


class GPUBackend(ABC):
    def __init__(self, device: int = 0):
        self.device = device

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool: ...

    @abstractmethod
    def get_usage(self) -> tuple[float, float]: ...
