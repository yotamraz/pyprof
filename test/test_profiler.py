from __future__ import annotations

import time

from pyprof.profiler import Profiler


def test_profiler_disables_gpu_when_backend_is_unavailable(monkeypatch) -> None:
    class FailingGPU:
        def __init__(self, pid: int, device: int) -> None:
            _ = pid
            _ = device
            raise RuntimeError("simulated missing backend")

        @classmethod
        def get_unavailable_reason(cls) -> str:
            return "simulated missing backend"

    monkeypatch.setattr("pyprof.profiler.ProfileGPU", FailingGPU)

    with Profiler(gpu=True, sleep_time=0.001) as profiler:
        time.sleep(0.01)

    data = profiler.get_recorded_data()
    peak_cpu, peak_ram, peak_gpu, peak_gpu_mem = profiler.get_peak_metrics()
    assert len(data) > 0
    assert "GPU_utilization_%" not in data.columns
    assert peak_cpu >= 0
    assert peak_ram >= 0
    assert peak_gpu is None
    assert peak_gpu_mem is None


def test_profiler_records_gpu_metrics_when_backend_is_available(monkeypatch) -> None:
    class DummyGPU:
        def __init__(self, pid: int, device: int) -> None:
            _ = pid
            _ = device
            self.backend_name = "dummy-gpu"

        @classmethod
        def get_unavailable_reason(cls) -> None:
            return None

        def get_usage(self) -> tuple[float, float]:
            return 12.5, 256.0

    monkeypatch.setattr("pyprof.profiler.ProfileGPU", DummyGPU)

    with Profiler(gpu=True, sleep_time=0.001) as profiler:
        time.sleep(0.01)

    data = profiler.get_recorded_data()
    peak_cpu, peak_ram, peak_gpu, peak_gpu_mem = profiler.get_peak_metrics()
    assert len(data) > 0
    assert "GPU_utilization_%" in data.columns
    assert "total_GPU_memory_usage_MB" in data.columns
    assert peak_cpu >= 0
    assert peak_ram >= 0
    assert peak_gpu is not None and peak_gpu >= 12.5
    assert peak_gpu_mem is not None and peak_gpu_mem >= 256.0
