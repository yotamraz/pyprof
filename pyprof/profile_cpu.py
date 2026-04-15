import psutil


class ProfileCPU:
    def __init__(self, pid: int) -> None:
        self.__pid = pid
        # psutil.cpu_count can return None in constrained environments.
        self.__cpu_count = psutil.cpu_count() or 1
        self.cpu_handler = psutil.Process(self.__pid)
        self.init_measure = self.cpu_handler.cpu_percent()

    def get_usage(self) -> float:
        return float(self.cpu_handler.cpu_percent() / self.__cpu_count)

