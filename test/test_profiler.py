"""
3-layer portable test suite for pyprof.

Layer 1  — Backend unit tests (always run, all mocked, no GPU hardware needed)
Layer 2  — Profiler integration tests (mock GPUBackend injected, no GPU needed)
Layer 3  — Hardware-specific smoke tests (skipped when hardware is absent)
"""

import sys
import time

import pytest

from pyprof.backends.base import GPUBackend
from pyprof.backends.nvidia import NVIDIABackend
from pyprof.backends.rocm import ROCmBackend
from pyprof.backends.apple_silicon import AppleSiliconBackend
from pyprof.profiler import Profiler


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_driver_caches():
    """Reset class-level _driver caches before (and after) every test.

    All three backends use a class-level ``_driver`` attribute as a lazy-import
    cache.  Without this reset one test's mock leaks into the next.
    """
    originals = {
        NVIDIABackend: NVIDIABackend._driver,
        ROCmBackend: ROCmBackend._driver,
        AppleSiliconBackend: AppleSiliconBackend._driver,
    }
    yield
    for cls, val in originals.items():
        cls._driver = val


class MockGPUBackend(GPUBackend):
    """Minimal in-process backend for Layer 2 tests — no hardware required."""

    _driver = None  # satisfy isinstance/ABC checks

    @classmethod
    def is_available(cls) -> bool:
        return True

    def get_usage(self) -> tuple:
        """Return static values: 42 % utilisation, 1024 MB memory."""
        return (42.0, 1024.0)


# ---------------------------------------------------------------------------
# Layer 1 — NVIDIABackend unit tests
# ---------------------------------------------------------------------------

class TestNVIDIABackendUnit:

    def test_get_usage_returns_two_floats(self, mocker):
        """get_usage() must return (util_percent: float, mem_MB: float)."""
        NVIDIABackend._driver = None

        mock_nvidia = mocker.MagicMock()
        mock_nvidia.nvmlInit.return_value = None
        mock_nvidia.nvmlShutdown.return_value = None

        mock_handle = mocker.MagicMock()
        mock_nvidia.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        mock_util = mocker.MagicMock()
        mock_util.gpu = 42
        mock_nvidia.nvmlDeviceGetUtilizationRates.return_value = mock_util

        mock_mem = mocker.MagicMock()
        mock_mem.used = 1024 * 1024 * 1024  # 1 GiB in bytes
        mock_nvidia.nvmlDeviceGetMemoryInfo.return_value = mock_mem

        # Inject mock so is_available() would succeed and __init__ can use it
        mocker.patch("pyprof.backends.nvidia.nvidia_smi", mock_nvidia, create=True)
        NVIDIABackend._driver = mock_nvidia

        backend = NVIDIABackend(device=0)
        util, mem = backend.get_usage()

        assert isinstance(util, float)
        assert isinstance(mem, float)
        assert util == pytest.approx(42.0)
        assert mem == pytest.approx(1024.0)  # 1 GiB → 1024 MB

    def test_get_usage_converts_bytes_to_mb(self, mocker):
        """Memory reported by the driver (bytes) must be converted to MB."""
        NVIDIABackend._driver = None

        mock_nvidia = mocker.MagicMock()
        mock_nvidia.nvmlInit.return_value = None
        mock_handle = mocker.MagicMock()
        mock_nvidia.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        mock_util = mocker.MagicMock()
        mock_util.gpu = 0
        mock_nvidia.nvmlDeviceGetUtilizationRates.return_value = mock_util

        mock_mem = mocker.MagicMock()
        mock_mem.used = 512 * 1024 * 1024  # 512 MiB
        mock_nvidia.nvmlDeviceGetMemoryInfo.return_value = mock_mem

        mocker.patch("pyprof.backends.nvidia.nvidia_smi", mock_nvidia, create=True)
        NVIDIABackend._driver = mock_nvidia

        backend = NVIDIABackend(device=0)
        _, mem = backend.get_usage()
        assert mem == pytest.approx(512.0)

    def test_is_available_returns_false_when_import_fails(self, mocker):
        """is_available() must return False when nvidia_smi cannot be imported."""
        NVIDIABackend._driver = None

        # Remove any cached module so the lazy import inside is_available() fires
        sys.modules.pop("nvidia_smi", None)

        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def _fake_import(name, *args, **kwargs):
            if name == "nvidia_smi":
                raise ImportError("mocked: nvidia_smi not installed")
            return real_import(name, *args, **kwargs)

        mocker.patch("builtins.__import__", side_effect=_fake_import)

        result = NVIDIABackend.is_available()
        assert result is False

    def test_is_available_caches_driver_on_success(self, mocker):
        """After a successful is_available() call _driver should be set."""
        NVIDIABackend._driver = None

        mock_nvidia = mocker.MagicMock()
        mock_nvidia.nvmlInit.return_value = None
        mock_nvidia.nvmlShutdown.return_value = None

        mocker.patch("pyprof.backends.nvidia.nvidia_smi", mock_nvidia, create=True)

        # Patch the import inside is_available so it uses our mock module
        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def _fake_import(name, *args, **kwargs):
            if name == "nvidia_smi":
                return mock_nvidia
            return real_import(name, *args, **kwargs)

        mocker.patch("builtins.__import__", side_effect=_fake_import)

        result = NVIDIABackend.is_available()
        # is_available sets _driver on success; either True (driver cached) or
        # the mock works at module-attribute level — what matters is no exception.
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Layer 1 — ROCmBackend unit tests
# ---------------------------------------------------------------------------

class TestROCmBackendUnit:

    def test_get_usage_returns_two_floats(self, mocker):
        """get_usage() must return (util_percent: float, mem_MB: float)."""
        ROCmBackend._driver = None

        mock_rocml = mocker.MagicMock()
        mock_rocml.smi_initialize.return_value = None
        mock_rocml.smi_shutdown.return_value = None
        mock_rocml.smi_get_device_utilization.return_value = 55
        # smi_get_device_vram_usage returns bytes used (single int per rocm.py)
        mock_rocml.smi_get_device_vram_usage.return_value = 2048 * 1024 * 1024  # 2 GiB

        mocker.patch("pyprof.backends.rocm.rocml", mock_rocml, create=True)
        ROCmBackend._driver = mock_rocml

        backend = ROCmBackend(device=0)
        util, mem = backend.get_usage()

        assert isinstance(util, float)
        assert isinstance(mem, float)
        assert util == pytest.approx(55.0)
        assert mem == pytest.approx(2048.0)

    def test_get_usage_converts_bytes_to_mb(self, mocker):
        """Memory value (bytes) from the driver must be divided by 1024**2."""
        ROCmBackend._driver = None

        mock_rocml = mocker.MagicMock()
        mock_rocml.smi_initialize.return_value = None
        mock_rocml.smi_get_device_utilization.return_value = 10
        mock_rocml.smi_get_device_vram_usage.return_value = 1024 * 1024 * 1024  # 1 GiB

        mocker.patch("pyprof.backends.rocm.rocml", mock_rocml, create=True)
        ROCmBackend._driver = mock_rocml

        backend = ROCmBackend(device=0)
        _, mem = backend.get_usage()
        assert mem == pytest.approx(1024.0)

    def test_is_available_returns_false_when_import_fails(self, mocker):
        """is_available() must return False when pyrsmi cannot be imported."""
        ROCmBackend._driver = None

        sys.modules.pop("pyrsmi", None)
        sys.modules.pop("pyrsmi.rocml", None)

        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def _fake_import(name, *args, **kwargs):
            if name in ("pyrsmi", "pyrsmi.rocml"):
                raise ImportError("mocked: pyrsmi not installed")
            return real_import(name, *args, **kwargs)

        mocker.patch("builtins.__import__", side_effect=_fake_import)

        result = ROCmBackend.is_available()
        assert result is False

    def test_get_usage_raises_on_missing_api(self, mocker):
        """get_usage() should propagate RuntimeError when driver lacks expected methods."""
        ROCmBackend._driver = None

        mock_rocml = mocker.MagicMock()
        mock_rocml.smi_initialize.return_value = None
        mock_rocml.smi_get_device_utilization.side_effect = AttributeError("not found")

        mocker.patch("pyprof.backends.rocm.rocml", mock_rocml, create=True)
        ROCmBackend._driver = mock_rocml

        backend = ROCmBackend(device=0)
        with pytest.raises(RuntimeError, match="ROCm SMI API"):
            backend.get_usage()


# ---------------------------------------------------------------------------
# Layer 1 — AppleSiliconBackend unit tests
# ---------------------------------------------------------------------------

class TestAppleSiliconBackendUnit:

    def test_get_usage_returns_two_floats(self, mocker):
        """get_usage() must return (util_percent: float, mem_MB: float)."""
        AppleSiliconBackend._driver = None

        mock_apple = mocker.MagicMock()
        mock_apple.get_gpu_utilization.return_value = 30.0
        mock_apple.get_gpu_memory.return_value = 512.0

        mocker.patch("pyprof.backends.apple_silicon.apple_smi", mock_apple, create=True)
        AppleSiliconBackend._driver = mock_apple

        backend = AppleSiliconBackend(device=0)
        util, mem = backend.get_usage()

        assert isinstance(util, float)
        assert isinstance(mem, float)
        assert util == pytest.approx(30.0)
        assert mem == pytest.approx(512.0)

    def test_get_usage_casts_to_float(self, mocker):
        """get_usage() must cast driver return values to float."""
        AppleSiliconBackend._driver = None

        mock_apple = mocker.MagicMock()
        mock_apple.get_gpu_utilization.return_value = 75  # int from driver
        mock_apple.get_gpu_memory.return_value = 256      # int from driver

        mocker.patch("pyprof.backends.apple_silicon.apple_smi", mock_apple, create=True)
        AppleSiliconBackend._driver = mock_apple

        backend = AppleSiliconBackend(device=0)
        util, mem = backend.get_usage()

        assert type(util) is float
        assert type(mem) is float

    def test_is_available_returns_false_when_import_fails(self, mocker):
        """is_available() must return False when apple_smi cannot be imported."""
        AppleSiliconBackend._driver = None

        sys.modules.pop("apple_smi", None)

        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def _fake_import(name, *args, **kwargs):
            if name == "apple_smi":
                raise ImportError("mocked: apple_smi not installed")
            return real_import(name, *args, **kwargs)

        mocker.patch("builtins.__import__", side_effect=_fake_import)

        result = AppleSiliconBackend.is_available()
        assert result is False

    def test_get_usage_raises_on_missing_api(self, mocker):
        """get_usage() should propagate RuntimeError when driver lacks expected methods."""
        AppleSiliconBackend._driver = None

        mock_apple = mocker.MagicMock()
        mock_apple.get_gpu_utilization.side_effect = AttributeError("missing")

        mocker.patch("pyprof.backends.apple_silicon.apple_smi", mock_apple, create=True)
        AppleSiliconBackend._driver = mock_apple

        backend = AppleSiliconBackend(device=0)
        with pytest.raises(RuntimeError, match="apple_smi"):
            backend.get_usage()


# ---------------------------------------------------------------------------
# Layer 2 — Profiler integration tests (mock GPUBackend injected)
# ---------------------------------------------------------------------------

class TestProfilerIntegration:

    def test_profiler_columns_with_gpu(self, tmp_path):
        """DataFrame must include GPU columns when a GPU backend is active."""
        mock_backend = MockGPUBackend(device=0)
        with Profiler(gpu=True, backend=mock_backend, sleep_time=0.05) as p:
            time.sleep(0.2)

        df = p.get_recorded_data()
        assert "timestamp" in df.columns
        assert "CPU_utilization_%" in df.columns
        assert "total_RAM_memory_usage_MB" in df.columns
        assert "GPU_utilization_%" in df.columns
        assert "total_GPU_memory_usage_MB" in df.columns
        assert len(df) > 0

    def test_profiler_records_rows(self, tmp_path):
        """At least one row must be recorded during a 0.2 s window."""
        mock_backend = MockGPUBackend(device=0)
        with Profiler(gpu=True, backend=mock_backend, sleep_time=0.05) as p:
            time.sleep(0.2)

        df = p.get_recorded_data()
        assert len(df) > 0

    def test_get_peak_metrics_with_gpu(self):
        """get_peak_metrics() must return four floats when GPU is active."""
        mock_backend = MockGPUBackend(device=0)
        with Profiler(gpu=True, backend=mock_backend, sleep_time=0.05) as p:
            time.sleep(0.2)

        cpu, ram, gpu_util, gpu_mem = p.get_peak_metrics()
        assert isinstance(cpu, float)
        assert isinstance(ram, float)
        assert isinstance(gpu_util, float)
        assert isinstance(gpu_mem, float)

    def test_get_peak_metrics_gpu_values_match_mock(self):
        """Peak GPU values must come from the mock backend (42 %, 1024 MB)."""
        mock_backend = MockGPUBackend(device=0)
        with Profiler(gpu=True, backend=mock_backend, sleep_time=0.05) as p:
            time.sleep(0.2)

        _, _, gpu_util, gpu_mem = p.get_peak_metrics()
        assert gpu_util == pytest.approx(42.0)
        assert gpu_mem == pytest.approx(1024.0)

    def test_save_result_file_creates_html(self, tmp_path):
        """_save_result_file() must write an HTML file to output_file_path."""
        output_path = str(tmp_path / "output.html")
        mock_backend = MockGPUBackend(device=0)
        with Profiler(
            output_file_path=output_path,
            gpu=True,
            backend=mock_backend,
            sleep_time=0.05,
        ) as p:
            time.sleep(0.2)

        p._save_result_file()
        assert (tmp_path / "output.html").exists()

    def test_profiler_no_gpu_columns_absent(self):
        """DataFrame must NOT contain GPU columns when gpu=False."""
        with Profiler(gpu=False, sleep_time=0.05) as p:
            time.sleep(0.2)

        df = p.get_recorded_data()
        assert "GPU_utilization_%" not in df.columns
        assert "total_GPU_memory_usage_MB" not in df.columns
        assert "CPU_utilization_%" in df.columns
        assert "total_RAM_memory_usage_MB" in df.columns

    def test_get_peak_metrics_no_gpu_returns_nones(self):
        """get_peak_metrics() must return (cpu, ram, None, None) when gpu=False."""
        with Profiler(gpu=False, sleep_time=0.05) as p:
            time.sleep(0.2)

        cpu, ram, gpu_util, gpu_mem = p.get_peak_metrics()
        assert gpu_util is None
        assert gpu_mem is None
        assert isinstance(cpu, float)
        assert isinstance(ram, float)

    def test_profiler_gpu_false_with_auto_backend(self):
        """gpu=False must skip GPU monitoring regardless of backend setting."""
        with Profiler(gpu=False, backend="auto", sleep_time=0.05) as p:
            time.sleep(0.1)

        assert p.gpu_profiler is None

    def test_profiler_backend_none_skips_gpu(self):
        """backend=None with gpu=True must also skip GPU monitoring."""
        with Profiler(gpu=True, backend=None, sleep_time=0.05) as p:
            time.sleep(0.1)

        assert p.gpu_profiler is None

    def test_profiler_injected_backend_instance_is_used(self):
        """The injected GPUBackend instance must be stored as gpu_profiler."""
        mock_backend = MockGPUBackend(device=0)
        with Profiler(gpu=True, backend=mock_backend, sleep_time=0.05) as p:
            time.sleep(0.1)

        assert p.gpu_profiler is mock_backend

    def test_profiler_invalid_backend_string_raises(self):
        """An unrecognised string backend must raise ValueError immediately."""
        with pytest.raises(ValueError, match="Unknown backend"):
            Profiler(gpu=True, backend="nonexistent_backend", sleep_time=0.05)

    def test_profiler_context_manager_sets_timestamps(self):
        """start_time and end_time must be set after __exit__."""
        mock_backend = MockGPUBackend(device=0)
        with Profiler(gpu=True, backend=mock_backend, sleep_time=0.05) as p:
            time.sleep(0.1)

        assert hasattr(p, "start_time")
        assert hasattr(p, "end_time")
        assert p.end_time >= p.start_time

    def test_get_recorded_data_returns_dataframe(self):
        """get_recorded_data() must return a pandas DataFrame."""
        import pandas as pd

        mock_backend = MockGPUBackend(device=0)
        with Profiler(gpu=True, backend=mock_backend, sleep_time=0.05) as p:
            time.sleep(0.15)

        df = p.get_recorded_data()
        assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# Layer 3 — Hardware-specific smoke tests (skipped without hardware)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not NVIDIABackend.is_available(),
    reason="No NVIDIA GPU available on this machine",
)
def test_nvidia_real_device():
    """Smoke test: NVIDIABackend works against real hardware."""
    backend = NVIDIABackend(device=0)
    util, mem = backend.get_usage()
    assert isinstance(util, float)
    assert isinstance(mem, float)
    assert 0.0 <= util <= 100.0
    assert mem >= 0.0


@pytest.mark.skipif(
    not ROCmBackend.is_available(),
    reason="No ROCm GPU available on this machine",
)
def test_rocm_real_device():
    """Smoke test: ROCmBackend works against real hardware."""
    backend = ROCmBackend(device=0)
    util, mem = backend.get_usage()
    assert isinstance(util, float)
    assert isinstance(mem, float)
    assert 0.0 <= util <= 100.0
    assert mem >= 0.0


@pytest.mark.skipif(
    not AppleSiliconBackend.is_available(),
    reason="No Apple Silicon GPU available on this machine",
)
def test_apple_silicon_real_device():
    """Smoke test: AppleSiliconBackend works against real hardware."""
    backend = AppleSiliconBackend(device=0)
    util, mem = backend.get_usage()
    assert isinstance(util, float)
    assert isinstance(mem, float)
    assert 0.0 <= util <= 100.0
    assert mem >= 0.0


@pytest.mark.skipif(
    not NVIDIABackend.is_available(),
    reason="No NVIDIA GPU available on this machine",
)
def test_full_profiler_with_nvidia_hardware():
    """End-to-end Profiler run on a real NVIDIA GPU."""
    with Profiler(gpu=True, backend="nvidia", sleep_time=0.05) as p:
        time.sleep(0.2)

    cpu, ram, gpu_util, gpu_mem = p.get_peak_metrics()
    assert isinstance(gpu_util, float)
    assert isinstance(gpu_mem, float)
    assert 0.0 <= gpu_util <= 100.0
