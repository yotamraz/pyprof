import amdsmi

from pyprof.profile_gpu_base import ProfileGPUBase


class ProfileGPUROCm(ProfileGPUBase):
    def __init__(self, device=0):
        amdsmi.amdsmi_init()
        handles = amdsmi.amdsmi_get_processor_handles()
        if device >= len(handles):
            raise RuntimeError(
                f"ROCm device {device} not found; only {len(handles)} device(s) available."
            )
        self._handle = handles[device]

    def get_usage(self) -> tuple[float, float]:
        activity = amdsmi.amdsmi_get_gpu_activity(self._handle)
        vram = amdsmi.amdsmi_get_gpu_vram_usage(self._handle)
        util = float(activity.get("gfx_activity", 0))
        memory_mb = float(vram.get("vram_used", 0))
        return util, memory_mb

