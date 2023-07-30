import os
import psutil

class ProfileRAM:
    def __init__(self, pid):
        self.__pid = pid
        self.ram_handler = psutil.Process(self.__pid)
        self.init_measure = self.ram_handler.memory_info().vms / 1024**2

    def get_usage(self):
        """

        :return:
        """
        return self.ram_handler.memory_info().vms / 1024**2