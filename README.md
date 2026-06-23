# pyprof

Track your CPU/RAM/GPU utilization with an easy-to-use context wrapper. Supports CUDA, AMD ROCm, and Apple Silicon (MPS) GPU backends.

## Install
```bash
# Base install (CPU + RAM profiling only)
pip install pyprof

# With NVIDIA CUDA GPU support
pip install "pyprof[cuda]"

# With AMD ROCm GPU support (requires ROCm >= 5.4 system-wide)
pip install "pyprof[rocm]"

# Development (adds pytest)
pip install "pyprof[dev]"
```

## Use

```python
from pyprof.profiler import Profiler

# Auto-detect GPU backend (CUDA → ROCm → MPS → none)
with Profiler(output_file_path='./output.html') as prof:
    funky_func()

# Explicit backend selection
with Profiler(output_file_path='./output.html', backend='cuda') as prof:
    funky_func()
```

## Results

#### stdout:

```stdout
########## CPU_utilization_% ##########
Peak: 74.425 %


########## total_RAM_memory_usage_MB ##########
Peak: 19519.48046875 MB


########## GPU_utilization_% ##########
Peak: 84.0 %


########## total_GPU_memory_usage_MB ##########
Peak: 7950.625 MB


########## total runtime ##########
0:00:04.035351
```

#### html:

![example_plot.png](doc/example_plot.png)
