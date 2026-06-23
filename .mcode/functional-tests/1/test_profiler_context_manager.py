"""
Functional tests for the Profiler context manager.

These tests verify that:
- Profiler(gpu=False) records CPU+RAM only (no GPU columns)
- Profiler(gpu=True) on CPU-only machine falls back to null GPU and records CPU+RAM only
- Profiler(backend='none') records CPU+RAM only
- get_recorded_data() returns a properly structured DataFrame
- get_peak_metrics() returns correct tuple shapes

All are target_only because the origin Profiler is not runnable on a CPU-only
machine (ImportError from module-level nvidia_smi import in profile_gpu.py).
"""
import pytest
import tempfile
import os
import warnings
from unittest.mock import patch, MagicMock

from pyprof.profiler import Profiler
from pyprof.profile_gpu_base import ProfileGPUBase
from pyprof.profile_gpu_null import ProfileGPUNull


@pytest.fixture
def tmp_html(tmp_path):
    """Provide a temporary HTML output path."""
    return str(tmp_path / "output.html")


class TestProfilerGpuFalse:
    """Profiler(gpu=False) — CPU+RAM only, no GPU columns."""

    def test_no_gpu_columns_in_dataframe(self, tmp_html):
        """When gpu=False, DataFrame has no GPU columns."""
        with Profiler(output_file_path=tmp_html, gpu=False, sleep_time=0.05) as prof:
            pass
        df = prof.get_recorded_data()
        assert 'GPU_utilization_%' not in df.columns
        assert 'total_GPU_memory_usage_MB' not in df.columns

    def test_has_cpu_and_ram_columns(self, tmp_html):
        """When gpu=False, DataFrame has CPU and RAM columns."""
        with Profiler(output_file_path=tmp_html, gpu=False, sleep_time=0.05) as prof:
            pass
        df = prof.get_recorded_data()
        assert 'CPU_utilization_%' in df.columns
        assert 'total_RAM_memory_usage_MB' in df.columns
        assert 'timestamp' in df.columns

    def test_has_rows_after_exit(self, tmp_html):
        """DataFrame contains rows after Profiler exits."""
        with Profiler(output_file_path=tmp_html, gpu=False, sleep_time=0.05) as prof:
            pass
        df = prof.get_recorded_data()
        assert len(df) > 0

    def test_cpu_values_are_nonnegative(self, tmp_html):
        """CPU utilization values are non-negative floats."""
        with Profiler(output_file_path=tmp_html, gpu=False, sleep_time=0.05) as prof:
            pass
        df = prof.get_recorded_data()
        assert (df['CPU_utilization_%'] >= 0).all()

    def test_ram_values_are_positive(self, tmp_html):
        """RAM memory values are positive (process uses some RAM)."""
        with Profiler(output_file_path=tmp_html, gpu=False, sleep_time=0.05) as prof:
            pass
        df = prof.get_recorded_data()
        assert (df['total_RAM_memory_usage_MB'] > 0).all()

    def test_gpu_false_uses_null_profiler(self, tmp_html):
        """When gpu=False, Profiler creates a ProfileGPUNull with backend='none'."""
        with Profiler(output_file_path=tmp_html, gpu=False, sleep_time=0.05) as prof:
            pass
        assert isinstance(prof.gpu_profiler, ProfileGPUNull)
        assert not prof.gpu_profiler


class TestProfilerGpuTrue:
    """Profiler(gpu=True) on CPU-only machine — auto-detects no GPU, falls back to null."""

    def test_gpu_true_auto_falls_back_to_null(self, tmp_html):
        """On CPU-only, Profiler(gpu=True) falls back to ProfileGPUNull."""
        with patch('importlib.util.find_spec', return_value=None), \
             patch('sys.platform', 'linux'), \
             patch('shutil.which', return_value=None):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with Profiler(output_file_path=tmp_html, gpu=True, sleep_time=0.05) as prof:
                    pass
        assert isinstance(prof.gpu_profiler, ProfileGPUNull)

    def test_gpu_true_no_gpu_columns_when_null(self, tmp_html):
        """When auto-fallback returns null, DataFrame has no GPU columns."""
        with patch('importlib.util.find_spec', return_value=None), \
             patch('sys.platform', 'linux'), \
             patch('shutil.which', return_value=None):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with Profiler(output_file_path=tmp_html, gpu=True, sleep_time=0.05) as prof:
                    pass
        df = prof.get_recorded_data()
        assert 'GPU_utilization_%' not in df.columns
        assert 'total_GPU_memory_usage_MB' not in df.columns


class TestProfilerBackendNone:
    """Profiler(backend='none') — explicitly disable GPU profiling."""

    def test_backend_none_no_gpu_columns(self, tmp_html):
        """Profiler with backend='none' produces no GPU columns."""
        with Profiler(output_file_path=tmp_html, gpu=True, backend='none', sleep_time=0.05) as prof:
            pass
        df = prof.get_recorded_data()
        assert 'GPU_utilization_%' not in df.columns
        assert 'total_GPU_memory_usage_MB' not in df.columns

    def test_backend_none_has_cpu_ram(self, tmp_html):
        """Profiler with backend='none' still collects CPU and RAM."""
        with Profiler(output_file_path=tmp_html, gpu=True, backend='none', sleep_time=0.05) as prof:
            pass
        df = prof.get_recorded_data()
        assert 'CPU_utilization_%' in df.columns
        assert 'total_RAM_memory_usage_MB' in df.columns
        assert len(df) > 0


class TestProfilerGetRecordedData:
    """Profiler.get_recorded_data() returns the correct DataFrame."""

    def test_returns_dataframe(self, tmp_html):
        """get_recorded_data returns a pandas DataFrame."""
        import pandas as pd
        with Profiler(output_file_path=tmp_html, gpu=False, sleep_time=0.05) as prof:
            pass
        df = prof.get_recorded_data()
        assert isinstance(df, pd.DataFrame)

    def test_dataframe_column_count_no_gpu(self, tmp_html):
        """Without GPU, DataFrame has exactly 3 columns: timestamp, cpu, ram."""
        with Profiler(output_file_path=tmp_html, gpu=False, sleep_time=0.05) as prof:
            pass
        df = prof.get_recorded_data()
        assert len(df.columns) == 3

    def test_dataframe_column_count_with_mock_gpu(self, tmp_html):
        """With GPU, DataFrame has exactly 5 columns including GPU columns."""
        mock_gpu = MagicMock()
        mock_gpu.get_usage.return_value = (50.0, 2048.0)
        mock_gpu.__bool__ = MagicMock(return_value=True)

        with patch('pyprof.profiler.create_gpu_profiler', return_value=mock_gpu):
            with Profiler(output_file_path=tmp_html, gpu=True, sleep_time=0.05) as prof:
                pass
        df = prof.get_recorded_data()
        assert len(df.columns) == 5

    def test_dataframe_column_names_no_gpu(self, tmp_html):
        """Without GPU, DataFrame column names are exactly timestamp, CPU, RAM."""
        with Profiler(output_file_path=tmp_html, gpu=False, sleep_time=0.05) as prof:
            pass
        df = prof.get_recorded_data()
        expected = {'timestamp', 'CPU_utilization_%', 'total_RAM_memory_usage_MB'}
        assert set(df.columns) == expected

    def test_dataframe_column_names_with_mock_gpu(self, tmp_html):
        """With GPU, DataFrame includes GPU_utilization_% and total_GPU_memory_usage_MB."""
        mock_gpu = MagicMock()
        mock_gpu.get_usage.return_value = (75.0, 4096.0)
        mock_gpu.__bool__ = MagicMock(return_value=True)

        with patch('pyprof.profiler.create_gpu_profiler', return_value=mock_gpu):
            with Profiler(output_file_path=tmp_html, gpu=True, sleep_time=0.05) as prof:
                pass
        df = prof.get_recorded_data()
        assert 'GPU_utilization_%' in df.columns
        assert 'total_GPU_memory_usage_MB' in df.columns


class TestProfilerGetPeakMetrics:
    """Profiler.get_peak_metrics() returns correct (cpu, ram, gpu_util, gpu_mem) tuple."""

    def test_peak_metrics_no_gpu_returns_none_for_gpu(self, tmp_html):
        """get_peak_metrics returns None for both GPU fields when no GPU."""
        with Profiler(output_file_path=tmp_html, gpu=False, sleep_time=0.05) as prof:
            pass
        cpu, ram, gpu_util, gpu_mem = prof.get_peak_metrics()
        assert gpu_util is None
        assert gpu_mem is None

    def test_peak_metrics_no_gpu_cpu_is_nonnegative(self, tmp_html):
        """get_peak_metrics CPU value is a non-negative number."""
        with Profiler(output_file_path=tmp_html, gpu=False, sleep_time=0.05) as prof:
            pass
        cpu, ram, gpu_util, gpu_mem = prof.get_peak_metrics()
        assert cpu >= 0.0

    def test_peak_metrics_no_gpu_ram_is_positive(self, tmp_html):
        """get_peak_metrics RAM value is a positive number."""
        with Profiler(output_file_path=tmp_html, gpu=False, sleep_time=0.05) as prof:
            pass
        cpu, ram, gpu_util, gpu_mem = prof.get_peak_metrics()
        assert ram > 0.0

    def test_peak_metrics_with_mock_gpu_returns_values(self, tmp_html):
        """get_peak_metrics returns GPU values when GPU is active."""
        mock_gpu = MagicMock()
        mock_gpu.get_usage.return_value = (60.0, 3072.0)
        mock_gpu.__bool__ = MagicMock(return_value=True)

        with patch('pyprof.profiler.create_gpu_profiler', return_value=mock_gpu):
            with Profiler(output_file_path=tmp_html, gpu=True, sleep_time=0.05) as prof:
                pass
        cpu, ram, gpu_util, gpu_mem = prof.get_peak_metrics()
        assert gpu_util is not None
        assert gpu_mem is not None
        assert gpu_util >= 0.0
        assert gpu_mem >= 0.0

    def test_peak_metrics_returns_four_tuple(self, tmp_html):
        """get_peak_metrics always returns a 4-element tuple."""
        with Profiler(output_file_path=tmp_html, gpu=False, sleep_time=0.05) as prof:
            pass
        result = prof.get_peak_metrics()
        assert len(result) == 4


class TestProfilerHTMLOutput:
    """Profiler saves HTML output file."""

    def test_saves_html_file_on_exit(self, tmp_html):
        """Profiler writes HTML file when output_file_path is set."""
        with Profiler(output_file_path=tmp_html, gpu=False, sleep_time=0.05) as prof:
            pass
        prof._save_result_file()
        assert os.path.exists(tmp_html)

    def test_html_file_is_nonempty(self, tmp_html):
        """HTML output file contains data."""
        with Profiler(output_file_path=tmp_html, gpu=False, sleep_time=0.05) as prof:
            pass
        prof._save_result_file()
        assert os.path.getsize(tmp_html) > 0
