import psutil


class ProfileRAM:
    def __init__(self, pid: int) -> None:
        self.__pid = pid
        self.ram_handler = psutil.Process(self.__pid)
        self.init_measure = self.ram_handler.memory_info().rss / 1024**2

    def get_usage(self) -> float:
        """

        :return:
        """
        return float(self.ram_handler.memory_info().rss / 1024**2)