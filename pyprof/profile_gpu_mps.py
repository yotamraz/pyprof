import plistlib
import shutil
import subprocess
import warnings

import psutil

from pyprof.profile_gpu_base import ProfileGPUBase


class ProfileGPUMPS(ProfileGPUBase):
    def __init__(self, device=0):
        self._device = device
        self._sudo_available = False

        if not shutil.which('powermetrics'):
            warnings.warn(
                "powermetrics not found; GPU utilization unavailable. "
                "Memory will still be reported as process RSS.",
                UserWarning,
                stacklevel=2,
            )
            return

        result = subprocess.run(
            ['sudo', '-n', '/usr/bin/powermetrics', '--version'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            self._sudo_available = True
        else:
            warnings.warn(
                "sudo access to powermetrics is unavailable; "
                "GPU utilization will be reported as 0.0. "
                "Memory will still be reported as process RSS.",
                UserWarning,
                stacklevel=2,
            )

    def get_usage(self) -> tuple[float, float]:
        memory_mb = psutil.Process().memory_info().rss / 1024**2

        if not self._sudo_available:
            return 0.0, memory_mb

        try:
            result = subprocess.run(
                [
                    'sudo', '-n', '/usr/bin/powermetrics',
                    '-f', 'plist',
                    '-n', '1',
                    '--samplers', 'gpu_power',
                    '--timeout', '1000',
                ],
                capture_output=True,
                timeout=5,
            )
            data = plistlib.loads(result.stdout)
            gpu_list = data.get("GPU", [])
            if gpu_list:
                # Use the requested device index if available, otherwise first entry
                idx = min(self._device, len(gpu_list) - 1)
                gpu_entry = gpu_list[idx]
                # Key may vary across macOS versions; try common names
                util = 0.0
                for key in ("gpu_active_ratio", "active_ratio", "gpu_utilization"):
                    val = gpu_entry.get(key)
                    if val is not None:
                        util = val
                        break
                return float(util) * 100.0, memory_mb
        except Exception:
            pass

        return 0.0, memory_mb
