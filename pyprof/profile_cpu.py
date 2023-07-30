import os
import psutil

class ProfileCPU:
    def __init__(self, pid):
        self.__pid = pid
        self.__cpu_count = os.cpu_count()
        self.cpu_handler = psutil.Process(self.__pid)
        self.init_measure = self.cpu_handler.cpu_percent()

    def get_usage(self):
        return self.cpu_handler.cpu_percent() / self.__cpu_count

