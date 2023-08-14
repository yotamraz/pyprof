import os
import time
from threading import Thread
from datetime import datetime

import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from pyprof.profile_cpu import ProfileCPU
from pyprof.profile_gpu import ProfileGPU
from pyprof.profile_memory import ProfileRAM


class Profiler:
    """
    A simple profiling class capable of monitoring CPU/RAM/CUDA usage
    """

    def __init__(self, output_file_path=None, device=0, gpu=True, sleep_time=0.1):
        self.pid = os.getpid()
        self.output_file_path = output_file_path
        self.device = device
        self.sleep_time = sleep_time
        self.exit_event_loop = False

        # init
        self.cpu_profiler = ProfileCPU(pid=self.pid)
        self.ram_profiler = ProfileRAM(pid=self.pid)
        self.gpu_profiler = None
        if gpu:
            self.gpu_profiler = ProfileGPU(pid=self.pid, device=self.device)

        # init dataframe for result
        self.df_records = self._init_df()
        self.event_thread = Thread(target=self._event_loop)

    def __enter__(self):
        self.event_thread.start()
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb, export=False):
        self.exit_event_loop = True
        self.event_thread.join()
        self.end_time = datetime.now()
        self._finalize()
        if export:
            self._print_peak_results()
            self._save_result_file()

    def _init_df(self):
        """
        init a dataframe to store all recorded data
        :return:
        """
        columns = ["timestamp", "CPU_utilization_%", "total_RAM_memory_usage_MB"]
        if self.gpu_profiler is not None:
            columns.extend(["GPU_utilization_%", "total_GPU_memory_usage_MB"])
        return pd.DataFrame(columns=columns)

    def _print_peak_results(self):
        """
        print peak usage for each profiler
        :return:
        """

        print(f"########## CPU_utilization_% ##########")
        print(f"Peak: {self.df_records['CPU_utilization_%'].max()} %")
        print("")
        print("")

        print(f"########## total_RAM_memory_usage_MB ##########")
        print(f"Peak: {self.df_records['total_RAM_memory_usage_MB'].max()} MB")
        print("")
        print("")

        print(f"########## GPU_utilization_% ##########")
        print(f"Peak: {self.df_records['GPU_utilization_%'].max()} %")
        print("")
        print("")

        print(f"########## total_GPU_memory_usage_MB ##########")
        print(f"Peak: {self.df_records['total_GPU_memory_usage_MB'].max()} MB")
        print("")
        print("")

        print(f"########## total runtime ##########")
        print(f"{self.end_time - self.start_time}")
        print("")
        print("")

    def _save_result_file(self):
        """
        save recordings as an HTML file containing plotly graphs
        :return:
        """
        # draw percentage and MB plots
        fig = make_subplots(rows=1, cols=2)

        fig.add_trace(
            go.Scatter(x=self.df_records['timestamp'], y=self.df_records['CPU_utilization_%'],
                       mode='lines', name='CPU_utilization_%'), row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=self.df_records['timestamp'], y=self.df_records['total_RAM_memory_usage_MB'],
                       mode='lines', name='total_RAM_memory_usage_MB'), row=1, col=2
        )

        if self.gpu_profiler is not None:
            fig.add_trace(
                go.Scatter(x=self.df_records['timestamp'], y=self.df_records['GPU_utilization_%'],
                           mode='lines', name='GPU_utilization_%'), row=1, col=1
            )

            fig.add_trace(
                go.Scatter(x=self.df_records['timestamp'], y=self.df_records['total_GPU_memory_usage_MB'],
                           mode='lines', name='total_GPU_memory_usage_MB'), row=1, col=2
            )

        if self.output_file_path is None:
            fig.show()
        else:
            fig.write_html(self.output_file_path)

    def _record(self):
        """
        record data at a specific point in time
        :return:
        """
        timestamp = datetime.now()
        cpu_usage = self.cpu_profiler.get_usage()
        memory_usage = self.ram_profiler.get_usage()
        gpu_util, gpu_usage = 0.0, 0.0
        if self.gpu_profiler is not None:
            gpu_util, gpu_usage = self.gpu_profiler.get_usage()

        self.df_records.loc[len(self.df_records.index), :] = [
            timestamp,
            cpu_usage,
            memory_usage,
            gpu_util,
            gpu_usage,
        ]

    def _event_loop(self):
        """
        execute record() every sleep_time seconds
        :return:
        """
        while not self.exit_event_loop:
            # do staff
            self._record()
            time.sleep(self.sleep_time)

    def _finalize(self):
        for _ in range(10):
            self._record()
            time.sleep(self.sleep_time)

    def get_recorded_data(self):
        """
        get recordings dataframe
        :return:
        """
        return self.df_records

    def get_peak_metrics(self):
        """
        get all peak metrics
        """
        peak_cpu_util = self.df_records['CPU_utilization_%'].max()
        peak_memory_usage = self.df_records['total_RAM_memory_usage_MB'].max()
        peak_gpu_util = self.df_records['GPU_utilization_%'].max()
        peak_gpu_memory_usage = self.df_records['total_GPU_memory_usage_MB'].max()

        return peak_cpu_util, peak_memory_usage, peak_gpu_util, peak_gpu_memory_usage
