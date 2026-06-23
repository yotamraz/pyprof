"""
Functional tests for create_gpu_profiler() factory function.

All entities are target_only because the origin (main branch) cannot run on this
CPU-only machine (module-level nvidia_smi import causes ImportError at import time).
"""
import warnings
import pytest
from unittest.mock import patch, MagicMock

from pyprof.gpu_factory import create_gpu_profiler
from pyprof.profile_gpu_null import ProfileGPUNull


class TestFactoryBackendAuto:
    """create_gpu_profiler(backend='auto') on a CPU-only machine."""

    def test_auto_no_gpu_returns_null(self):
        """On CPU-only machine, auto-detection returns a ProfileGPUNull instance."""
        with patch('importlib.util.find_spec', return_value=None), \
             patch('sys.platform', 'linux'), \
             patch('shutil.which', return_value=None):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                result = create_gpu_profiler(backend='auto')
        assert isinstance(result, ProfileGPUNull)

    def test_auto_no_gpu_result_is_falsy(self):
        """ProfileGPUNull returned by auto fallback evaluates to False."""
        with patch('importlib.util.find_spec', return_value=None), \
             patch('sys.platform', 'linux'), \
             patch('shutil.which', return_value=None):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                result = create_gpu_profiler(backend='auto')
        assert not result
        assert bool(result) is False

    def test_auto_no_gpu_emits_user_warning(self):
        """Auto fallback to null backend emits a UserWarning."""
        with patch('importlib.util.find_spec', return_value=None), \
             patch('sys.platform', 'linux'), \
             patch('shutil.which', return_value=None):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = create_gpu_profiler(backend='auto')
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) >= 1

    def test_auto_null_get_usage_returns_zeros(self):
        """ProfileGPUNull from auto fallback returns (0.0, 0.0) from get_usage."""
        with patch('importlib.util.find_spec', return_value=None), \
             patch('sys.platform', 'linux'), \
             patch('shutil.which', return_value=None):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                result = create_gpu_profiler(backend='auto')
        util, mem = result.get_usage()
        assert util == 0.0
        assert mem == 0.0


class TestFactoryBackendNone:
    """create_gpu_profiler(backend='none') explicitly disables GPU profiling."""

    def test_none_returns_null_instance(self):
        """backend='none' returns a ProfileGPUNull instance."""
        result = create_gpu_profiler(backend='none')
        assert isinstance(result, ProfileGPUNull)

    def test_none_result_is_falsy(self):
        """ProfileGPUNull from backend='none' evaluates to False."""
        result = create_gpu_profiler(backend='none')
        assert not result

    def test_none_emits_no_user_warning(self):
        """backend='none' does NOT emit a UserWarning (warn=False)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = create_gpu_profiler(backend='none')
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_none_get_usage_returns_zeros(self):
        """ProfileGPUNull from backend='none' returns (0.0, 0.0) from get_usage."""
        result = create_gpu_profiler(backend='none')
        util, mem = result.get_usage()
        assert util == 0.0
        assert mem == 0.0


class TestFactoryExplicitCUDA:
    """create_gpu_profiler(backend='cuda') on machine without NVIDIA hardware."""

    def test_cuda_raises_runtime_error_without_hardware(self):
        """Explicit 'cuda' raises RuntimeError when nvidia-ml-py3 not installed."""
        with patch('importlib.util.find_spec', return_value=None):
            with pytest.raises(RuntimeError) as exc_info:
                create_gpu_profiler(backend='cuda')
        assert 'cuda' in str(exc_info.value).lower()

    def test_cuda_error_message_mentions_nvidia(self):
        """RuntimeError message mentions nvidia for helpful diagnostics."""
        with patch('importlib.util.find_spec', return_value=None):
            with pytest.raises(RuntimeError) as exc_info:
                create_gpu_profiler(backend='cuda')
        err = str(exc_info.value).lower()
        assert 'nvidia' in err or 'cuda' in err


class TestFactoryExplicitROCm:
    """create_gpu_profiler(backend='rocm') on machine without AMD hardware."""

    def test_rocm_raises_runtime_error_without_amdsmi(self):
        """Explicit 'rocm' raises RuntimeError when amdsmi not installed."""
        with patch('importlib.util.find_spec', return_value=None):
            with pytest.raises(RuntimeError) as exc_info:
                create_gpu_profiler(backend='rocm')
        assert 'rocm' in str(exc_info.value).lower()


class TestFactoryExplicitMPS:
    """create_gpu_profiler(backend='mps') on non-macOS machine."""

    def test_mps_raises_runtime_error_on_linux(self):
        """Explicit 'mps' raises RuntimeError on Linux (no powermetrics)."""
        with patch('sys.platform', 'linux'), patch('shutil.which', return_value=None):
            with pytest.raises(RuntimeError) as exc_info:
                create_gpu_profiler(backend='mps')
        assert 'mps' in str(exc_info.value).lower()


class TestFactoryUnknownBackend:
    """create_gpu_profiler with an unrecognized backend string."""

    def test_unknown_backend_raises_runtime_error(self):
        """Unrecognized backend string raises RuntimeError."""
        with pytest.raises(RuntimeError) as exc_info:
            create_gpu_profiler(backend='tpu')
        assert 'tpu' in str(exc_info.value).lower() or 'unknown' in str(exc_info.value).lower()

    def test_empty_string_backend_raises_runtime_error(self):
        """Empty backend string raises RuntimeError."""
        with pytest.raises(RuntimeError):
            create_gpu_profiler(backend='')
