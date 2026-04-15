import os
import time
import warnings
from threading import Thread
from datetime import datetime
from typing import Optional, cast

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

    def __init__(
        self,
        output_file_path: Optional[str] = None,
        device: int = 0,
        gpu: bool = True,
        sleep_time: float = 0.1,
    ) -> None:
        self.pid = os.getpid()
        self.output_file_path = output_file_path
        self.device = device
        self.sleep_time = sleep_time
        self.exit_event_loop = False
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.gpu_backend: Optional[str] = None
        self.gpu_unavailable_reason: Optional[str] = None

        # init profilers
        self.cpu_profiler = ProfileCPU(pid=self.pid)
        self.ram_profiler = ProfileRAM(pid=self.pid)
        self.gpu_profiler: Optional[ProfileGPU] = None
        if gpu:
            unavailable_reason = ProfileGPU.get_unavailable_reason()
            if unavailable_reason is None:
                self.gpu_profiler = ProfileGPU(pid=self.pid, device=self.device)
                self.gpu_backend = self.gpu_profiler.backend_name
            else:
                self.gpu_unavailable_reason = unavailable_reason
                warnings.warn(
                    f"GPU profiling disabled: {unavailable_reason}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # init dataframe for result
        self.df_records = self._init_df()
        self.event_thread = Thread(target=self._event_loop, daemon=True)

    def __enter__(self) -> "Profiler":
        self.event_thread.start()
        self.start_time = datetime.now()
        return self

    def __exit__(
        self, exc_type: object, exc_val: object, exc_tb: object, export: bool = False
    ) -> None:
        self.exit_event_loop = True
        self.event_thread.join()
        self.end_time = datetime.now()
        self._finalize()
        if export:
            self._print_peak_results()
            self._save_result_file()

    def _init_df(self) -> pd.DataFrame:
        """
        init a dataframe to store all recorded data
        :return:
        """
        columns = ["timestamp", "CPU_utilization_%", "total_RAM_memory_usage_MB"]
        if self.gpu_profiler is not None:
            columns.extend(["GPU_utilization_%", "total_GPU_memory_usage_MB"])
        return pd.DataFrame(columns=columns)

    def _print_peak_results(self) -> None:
        """
        print peak usage for each profiler
        :return:
        """

        print("########## CPU_utilization_% ##########")
        print(f"Peak: {self.df_records['CPU_utilization_%'].max()} %")
        print("")
        print("")

        print("########## total_RAM_memory_usage_MB ##########")
        print(f"Peak: {self.df_records['total_RAM_memory_usage_MB'].max()} MB")
        print("")
        print("")

        if self.gpu_profiler is not None:
            print("########## GPU_utilization_% ##########")
            print(f"Peak: {self.df_records['GPU_utilization_%'].max()} %")
            print("")
            print("")

            print("########## total_GPU_memory_usage_MB ##########")
            print(f"Peak: {self.df_records['total_GPU_memory_usage_MB'].max()} MB")
            print("")
            print("")
        else:
            print("########## GPU ##########")
            print("GPU profiling was disabled for this run.")
            if self.gpu_unavailable_reason is not None:
                print(f"Reason: {self.gpu_unavailable_reason}")
            print("")
            print("")

        print("########## total runtime ##########")
        if self.start_time is not None and self.end_time is not None:
            print(f"{self.end_time - self.start_time}")
        else:
            print("unknown")
        print("")
        print("")

    def _save_result_file(self) -> None:
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

    def _record(self) -> None:
        """
        record data at a specific point in time
        :return:
        """
        timestamp = datetime.now()
        cpu_usage = self.cpu_profiler.get_usage()
        memory_usage = self.ram_profiler.get_usage()
        row_data: list[object] = [timestamp, cpu_usage, memory_usage]
        if self.gpu_profiler is not None:
            gpu_util, gpu_usage = self.gpu_profiler.get_usage()
            row_data.extend([gpu_util, gpu_usage])

        self.df_records.loc[len(self.df_records.index), :] = row_data

    def _event_loop(self) -> None:
        """
        execute record() every sleep_time seconds
        :return:
        """
        while not self.exit_event_loop:
            # do staff
            self._record()
            time.sleep(self.sleep_time)

    def _finalize(self) -> None:
        for _ in range(10):
            self._record()
            time.sleep(self.sleep_time)

    def get_recorded_data(self) -> pd.DataFrame:
        """
        get recordings dataframe
        :return:
        """
        return self.df_records

    def get_peak_metrics(
        self,
    ) -> tuple[float, float, Optional[float], Optional[float]]:
        """
        get all peak metrics
        """
        peak_cpu_util = cast(float, self.df_records['CPU_utilization_%'].max())
        peak_memory_usage = cast(float, self.df_records['total_RAM_memory_usage_MB'].max())
        peak_gpu_util: Optional[float] = None
        peak_gpu_memory_usage: Optional[float] = None
        if self.gpu_profiler is not None:
            peak_gpu_util = cast(float, self.df_records['GPU_utilization_%'].max())
            peak_gpu_memory_usage = cast(
                float, self.df_records['total_GPU_memory_usage_MB'].max()
            )

        return peak_cpu_util, peak_memory_usage, peak_gpu_util, peak_gpu_memory_usage
