import pytest
import warnings
from unittest.mock import patch, MagicMock

from pyprof.profile_gpu_base import ProfileGPUBase
from pyprof.profile_gpu_null import ProfileGPUNull


# ─── ProfileGPUBase ─────────────────────────────────────────────────────────

class TestProfileGPUBase:
    def test_get_usage_raises_not_implemented(self):
        """Direct subclass without get_usage must raise TypeError at instantiation."""
        class Broken(ProfileGPUBase):
            pass
        with pytest.raises(TypeError):
            Broken()


# ─── ProfileGPUNull ─────────────────────────────────────────────────────────

class TestProfileGPUNull:
    def test_returns_zeros(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            null = ProfileGPUNull()
        util, mem = null.get_usage()
        assert util == 0.0
        assert mem == 0.0

    def test_emits_user_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ProfileGPUNull()
        assert any(issubclass(warning.category, UserWarning) for warning in w)


# ─── create_gpu_profiler ─────────────────────────────────────────────────────

class TestCreateGpuProfiler:
    def test_backend_none_returns_null(self):
        """backend='none' returns a falsy ProfileGPUNull without a UserWarning."""
        from pyprof.gpu_factory import create_gpu_profiler
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = create_gpu_profiler(backend='none')
        assert isinstance(result, ProfileGPUNull)
        assert not result
        assert not any(issubclass(x.category, UserWarning) for x in w)

    def test_auto_no_gpu_returns_null_with_warning(self):
        """When no GPU library is available and not darwin, factory returns ProfileGPUNull with warning."""
        from pyprof.gpu_factory import create_gpu_profiler
        with patch('importlib.util.find_spec', return_value=None), \
             patch('sys.platform', 'linux'), \
             patch('shutil.which', return_value=None):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = create_gpu_profiler(backend='auto')
        assert isinstance(result, ProfileGPUNull)
        assert not result
        assert any(issubclass(x.category, UserWarning) for x in w)

    def test_auto_cuda_available(self):
        """When nvidia_smi is detectable via find_spec, factory auto-detects CUDA and returns ProfileGPUCUDA."""
        from pyprof.gpu_factory import create_gpu_profiler
        mock_nvidia = MagicMock()
        mock_handler = MagicMock()
        mock_nvidia.nvmlDeviceGetHandleByIndex.return_value = mock_handler
        mock_nvidia.nvmlDeviceGetUtilizationRates.return_value = MagicMock(gpu=50.0)
        mock_nvidia.nvmlDeviceGetMemoryInfo.return_value = MagicMock(used=2048 * 1024**2)

        with patch.dict('sys.modules', {'nvidia_smi': mock_nvidia}), \
             patch('importlib.util.find_spec', return_value=MagicMock()):
            from importlib import reload
            import pyprof.profile_gpu_cuda as cuda_mod
            reload(cuda_mod)
            result = create_gpu_profiler(backend='auto')
            assert type(result).__name__ == 'ProfileGPUCUDA'

    def test_auto_rocm_available(self):
        """When amdsmi is in sys.modules (but nvidia_smi is not), factory returns ProfileGPUROCm."""
        from pyprof.gpu_factory import create_gpu_profiler
        mock_amdsmi = MagicMock()
        mock_handle = MagicMock()
        mock_amdsmi.amdsmi_get_processor_handles.return_value = [mock_handle]
        mock_amdsmi.amdsmi_get_gpu_activity.return_value = {"gfx_activity": 30.0}
        mock_amdsmi.amdsmi_get_gpu_vram_usage.return_value = {"vram_used": 1024.0}

        def fake_find_spec(name):
            return None if name == 'nvidia_smi' else MagicMock()

        with patch.dict('sys.modules', {'amdsmi': mock_amdsmi}), \
             patch('importlib.util.find_spec', side_effect=fake_find_spec):
            from importlib import reload
            import pyprof.profile_gpu_rocm as rocm_mod
            reload(rocm_mod)
            result = create_gpu_profiler(backend='auto')
            assert type(result).__name__ == 'ProfileGPUROCm'

    def test_auto_mps_available(self):
        """On darwin with powermetrics available, factory auto-detects MPS and returns ProfileGPUMPS."""
        from pyprof.gpu_factory import create_gpu_profiler

        with patch('importlib.util.find_spec', return_value=None), \
             patch('sys.platform', 'darwin'), \
             patch('shutil.which', return_value='/usr/bin/powermetrics'), \
             patch('subprocess.run', return_value=MagicMock(returncode=1)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                result = create_gpu_profiler(backend='auto')

        from pyprof.profile_gpu_mps import ProfileGPUMPS
        assert isinstance(result, ProfileGPUMPS)

    def test_explicit_cuda_unavailable_raises(self):
        """Explicit 'cuda' when nvidia_smi not found raises RuntimeError."""
        from pyprof.gpu_factory import create_gpu_profiler
        with patch('importlib.util.find_spec', return_value=None):
            with pytest.raises(RuntimeError, match="cuda"):
                create_gpu_profiler(backend='cuda')

    def test_explicit_rocm_unavailable_raises(self):
        """Explicit 'rocm' when amdsmi not found raises RuntimeError."""
        from pyprof.gpu_factory import create_gpu_profiler
        with patch('importlib.util.find_spec', return_value=None):
            with pytest.raises(RuntimeError, match="rocm"):
                create_gpu_profiler(backend='rocm')

    def test_explicit_mps_unavailable_raises(self):
        """Explicit 'mps' on non-darwin raises RuntimeError."""
        from pyprof.gpu_factory import create_gpu_profiler
        with patch('sys.platform', 'linux'), patch('shutil.which', return_value=None):
            with pytest.raises(RuntimeError, match="mps"):
                create_gpu_profiler(backend='mps')

    def test_unknown_backend_raises(self):
        from pyprof.gpu_factory import create_gpu_profiler
        with pytest.raises(RuntimeError, match="Unknown backend"):
            create_gpu_profiler(backend='invalid_backend')


# ─── Profiler with mocked GPU profiler ─────────────────────────────────────

class TestProfilerWithMockedGPU:
    """Test Profiler context manager lifecycle with a mock GPU profiler."""

    def test_profiler_no_gpu(self, tmp_path):
        """Profiler runs without GPU (backend='none')."""
        from pyprof.profiler import Profiler
        output_file = tmp_path / "output.html"

        with Profiler(output_file_path=str(output_file), gpu=False, sleep_time=0.05) as prof:
            pass

        df = prof.get_recorded_data()
        assert len(df) > 0
        assert 'CPU_utilization_%' in df.columns
        assert 'total_RAM_memory_usage_MB' in df.columns
        assert 'GPU_utilization_%' not in df.columns

    def test_profiler_with_mock_gpu(self, tmp_path):
        """Profiler runs with a mocked GPU profiler."""
        from pyprof.profiler import Profiler
        from pyprof.profile_gpu_base import ProfileGPUBase
        output_file = tmp_path / "output.html"

        mock_gpu = MagicMock(spec=ProfileGPUBase)
        mock_gpu.get_usage.return_value = (50.0, 2048.0)

        with patch('pyprof.profiler.create_gpu_profiler', return_value=mock_gpu):
            with Profiler(output_file_path=str(output_file), gpu=True, sleep_time=0.05) as prof:
                pass

        df = prof.get_recorded_data()
        assert len(df) > 0
        assert 'GPU_utilization_%' in df.columns
        assert 'total_GPU_memory_usage_MB' in df.columns
        assert df['GPU_utilization_%'].iloc[0] == 50.0
        assert df['total_GPU_memory_usage_MB'].iloc[0] == 2048.0

    def test_profiler_get_peak_metrics_no_gpu(self, tmp_path):
        """get_peak_metrics returns None for GPU metrics when no GPU."""
        from pyprof.profiler import Profiler
        output_file = tmp_path / "output.html"

        with Profiler(output_file_path=str(output_file), gpu=False, sleep_time=0.05) as prof:
            pass

        peak_cpu, peak_ram, peak_gpu_util, peak_gpu_mem = prof.get_peak_metrics()
        assert peak_cpu is not None
        assert peak_ram is not None
        assert peak_gpu_util is None
        assert peak_gpu_mem is None

    def test_profiler_get_peak_metrics_with_gpu(self, tmp_path):
        """get_peak_metrics returns GPU metrics when GPU is present."""
        from pyprof.profiler import Profiler
        from pyprof.profile_gpu_base import ProfileGPUBase
        output_file = tmp_path / "output.html"

        mock_gpu = MagicMock(spec=ProfileGPUBase)
        mock_gpu.get_usage.return_value = (75.0, 4096.0)

        with patch('pyprof.profiler.create_gpu_profiler', return_value=mock_gpu):
            with Profiler(output_file_path=str(output_file), gpu=True, sleep_time=0.05) as prof:
                pass

        peak_cpu, peak_ram, peak_gpu_util, peak_gpu_mem = prof.get_peak_metrics()
        assert peak_gpu_util == 75.0
        assert peak_gpu_mem == 4096.0

    def test_profiler_saves_html(self, tmp_path):
        """Profiler saves an HTML file when output_file_path is set."""
        from pyprof.profiler import Profiler
        output_file = tmp_path / "output.html"

        with Profiler(output_file_path=str(output_file), gpu=False, sleep_time=0.05) as prof:
            pass
        prof._save_result_file()

        assert output_file.exists()

    def test_profiler_backend_parameter_passed_to_factory(self, tmp_path):
        """Profiler backend parameter is correctly passed to create_gpu_profiler."""
        from pyprof.profiler import Profiler
        output_file = tmp_path / "output.html"

        with patch('pyprof.profiler.create_gpu_profiler', return_value=None) as mock_factory:
            with Profiler(output_file_path=str(output_file), gpu=True, backend='cuda', sleep_time=0.05) as prof:
                pass

        mock_factory.assert_called_once_with(backend='cuda', device=0)

    def test_profiler_gpu_false_passes_none_backend(self, tmp_path):
        """When gpu=False, create_gpu_profiler is called with backend='none'."""
        from pyprof.profiler import Profiler
        output_file = tmp_path / "output.html"

        with patch('pyprof.profiler.create_gpu_profiler', return_value=None) as mock_factory:
            with Profiler(output_file_path=str(output_file), gpu=False, sleep_time=0.05) as prof:
                pass

        mock_factory.assert_called_once_with(backend='none', device=0)


# ─── MockGPU backends ────────────────────────────────────────────────────────

class TestProfileGPUCUDAMocked:
    """Test ProfileGPUCUDA with mocked nvidia_smi."""

    def test_get_usage_returns_tuple(self):
        """ProfileGPUCUDA.get_usage returns (utilization, memory_MB)."""
        mock_nvidia = MagicMock()
        mock_handler = MagicMock()
        mock_util = MagicMock()
        mock_util.gpu = 42.0
        mock_mem = MagicMock()
        mock_mem.used = 2048 * 1024**2
        mock_nvidia.nvmlDeviceGetHandleByIndex.return_value = mock_handler
        mock_nvidia.nvmlDeviceGetUtilizationRates.return_value = mock_util
        mock_nvidia.nvmlDeviceGetMemoryInfo.return_value = mock_mem

        with patch.dict('sys.modules', {'nvidia_smi': mock_nvidia}):
            from importlib import reload
            import pyprof.profile_gpu_cuda as mod
            reload(mod)
            cuda = mod.ProfileGPUCUDA(device=0)

        util, mem = cuda.get_usage()
        assert util == 42.0
        assert mem == pytest.approx(2048.0)


class TestProfileGPUROCmMocked:
    """Test ProfileGPUROCm with mocked amdsmi."""

    def test_get_usage_returns_tuple(self):
        """ProfileGPUROCm.get_usage returns (utilization, memory_MB)."""
        mock_amdsmi = MagicMock()
        mock_handle = MagicMock()
        mock_amdsmi.amdsmi_get_processor_handles.return_value = [mock_handle]
        mock_amdsmi.amdsmi_get_gpu_activity.return_value = {"gfx_activity": 55.0}
        mock_amdsmi.amdsmi_get_gpu_vram_usage.return_value = {"vram_used": 1024.0}

        with patch.dict('sys.modules', {'amdsmi': mock_amdsmi}):
            from importlib import reload
            import pyprof.profile_gpu_rocm as mod
            reload(mod)
            rocm = mod.ProfileGPUROCm(device=0)

        util, mem = rocm.get_usage()
        assert util == 55.0
        assert mem == 1024.0


class TestProfileGPUMPSMocked:
    """Test ProfileGPUMPS with mocked subprocess."""

    def test_get_usage_no_sudo(self):
        """ProfileGPUMPS returns 0.0 utilization with memory when sudo unavailable."""
        import pyprof.profile_gpu_mps as mod

        with patch('shutil.which', return_value='/usr/bin/powermetrics'), \
             patch('subprocess.run') as mock_run, \
             patch('psutil.Process') as mock_psutil:
            # Make sudo -n check fail (no sudo)
            mock_run.return_value = MagicMock(returncode=1)
            mock_proc = MagicMock()
            mock_proc.memory_info.return_value.rss = 512 * 1024**2
            mock_psutil.return_value = mock_proc

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                mps = mod.ProfileGPUMPS(device=0)

        with patch('psutil.Process') as mock_psutil2:
            mock_proc2 = MagicMock()
            mock_proc2.memory_info.return_value.rss = 512 * 1024**2
            mock_psutil2.return_value = mock_proc2
            util, mem = mps.get_usage()

        assert util == 0.0
        assert mem == pytest.approx(512.0)

    def test_get_usage_with_sudo_parses_plist(self):
        """ProfileGPUMPS parses powermetrics plist and returns utilization when sudo is available."""
        import plistlib
        import pyprof.profile_gpu_mps as mod

        fake_plist = {"GPU": [{"gpu_active_ratio": 0.42}]}
        fake_plist_bytes = plistlib.dumps(fake_plist)

        with patch('shutil.which', return_value='/usr/bin/powermetrics'), \
             patch('subprocess.run') as mock_run, \
             patch('psutil.Process') as mock_psutil:
            # Init: sudo -n check passes; get_usage: powermetrics returns plist bytes
            mock_run.side_effect = [
                MagicMock(returncode=0),
                MagicMock(returncode=0, stdout=fake_plist_bytes),
            ]
            mock_proc = MagicMock()
            mock_proc.memory_info.return_value.rss = 1024 * 1024**2
            mock_psutil.return_value = mock_proc

            mps = mod.ProfileGPUMPS(device=0)
            util, mem = mps.get_usage()

        assert util == pytest.approx(42.0)  # 0.42 * 100
        assert mem == pytest.approx(1024.0)

    def test_get_usage_zero_utilization_not_overridden(self):
        """gpu_active_ratio=0.0 must not fall through to alternative keys via or-chain."""
        import plistlib
        import pyprof.profile_gpu_mps as mod

        fake_plist = {"GPU": [{"gpu_active_ratio": 0.0, "active_ratio": 0.5}]}
        fake_plist_bytes = plistlib.dumps(fake_plist)

        with patch('shutil.which', return_value='/usr/bin/powermetrics'), \
             patch('subprocess.run') as mock_run, \
             patch('psutil.Process') as mock_psutil:
            mock_run.side_effect = [
                MagicMock(returncode=0),
                MagicMock(returncode=0, stdout=fake_plist_bytes),
            ]
            mock_proc = MagicMock()
            mock_proc.memory_info.return_value.rss = 512 * 1024**2
            mock_psutil.return_value = mock_proc

            mps = mod.ProfileGPUMPS(device=0)
            util, mem = mps.get_usage()

        assert util == pytest.approx(0.0)  # gpu_active_ratio=0.0 must win, not active_ratio=0.5
        assert mem == pytest.approx(512.0)


# ─── Hardware integration tests (skipped by default) ────────────────────────

@pytest.mark.cuda
class TestProfileGPUCUDAIntegration:
    """Integration tests requiring a real NVIDIA GPU."""

    def test_get_usage_returns_nonnegative(self):
        pytest.importorskip("nvidia_smi", reason="nvidia-ml-py3 not installed")
        from pyprof.profile_gpu_cuda import ProfileGPUCUDA
        cuda = ProfileGPUCUDA(device=0)
        util, mem = cuda.get_usage()
        assert util >= 0.0
        assert mem >= 0.0


@pytest.mark.rocm
class TestProfileGPUROCmIntegration:
    """Integration tests requiring a real AMD ROCm GPU."""

    def test_get_usage_returns_nonnegative(self):
        pytest.importorskip("amdsmi", reason="amdsmi not installed")
        from pyprof.profile_gpu_rocm import ProfileGPUROCm
        rocm = ProfileGPUROCm(device=0)
        util, mem = rocm.get_usage()
        assert util >= 0.0
        assert mem >= 0.0


@pytest.mark.mps
class TestProfileGPUMPSIntegration:
    """Integration tests requiring Apple Silicon."""

    def test_get_usage_returns_nonnegative(self):
        import sys
        import shutil
        if sys.platform != 'darwin' or not shutil.which('powermetrics'):
            pytest.skip("Not on Apple Silicon or powermetrics not found")
        from pyprof.profile_gpu_mps import ProfileGPUMPS
        mps = ProfileGPUMPS(device=0)
        util, mem = mps.get_usage()
        assert util >= 0.0
        assert mem >= 0.0
