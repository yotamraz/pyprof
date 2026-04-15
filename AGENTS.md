# AGENTS.md

## Cursor Cloud specific instructions

### Overview

**pyprof** is a lightweight Python profiling library that monitors CPU, RAM, and GPU (CUDA) utilization using a context-manager API (`Profiler`). It records metrics into a pandas DataFrame and can export results as Plotly HTML charts.

### Python Version

The project targets Python >= 3.8. The pinned dependencies in `requirements.txt` (especially `pandas==1.5.3` and `numpy==1.23.5`) are **not compatible with Python 3.12+**. Use Python 3.11 (installed via `deadsnakes` PPA). A virtualenv is set up at `/workspace/.venv`.

### Development Setup

```bash
source /workspace/.venv/bin/activate
```

All commands below assume the venv is activated.

### Key Gotchas

1. **GPU module crashes at import time without NVIDIA drivers.** `pyprof/profile_gpu.py` calls `nvidia_smi.nvmlInit()` at module level. To import `pyprof.profiler` in a no-GPU environment, mock `nvidia_smi` before importing:
   ```python
   import sys, types
   mock = types.ModuleType('nvidia_smi')
   mock.nvmlInit = lambda: None
   mock.nvmlDeviceGetHandleByIndex = lambda d: type('H', (), {})()
   mock.nvmlDeviceGetUtilizationRates = lambda h: type('U', (), {'gpu': 0.0})()
   mock.nvmlDeviceGetMemoryInfo = lambda h: type('M', (), {'used': 0})()
   sys.modules['nvidia_smi'] = mock
   ```

2. **`gpu=False` bug.** When `Profiler(gpu=False)`, the `_record()` method still writes 5 values (including GPU placeholders) to a 3-column DataFrame, causing a `ValueError`. Use `gpu=True` with the mock above, or fix the upstream bug.

3. **`__exit__` does not export by default.** The `export` parameter defaults to `False`, so `_print_peak_results()` and `_save_result_file()` must be called manually after the context block if you want output.

4. **Existing test (`test/test_profiler.py`) requires CUDA GPU + OpenCV + PyTorch.** These are not in `requirements.txt` and the test has hardcoded local paths. It will not pass in Cloud Agent environments.

### Commands

| Task   | Command |
|--------|---------|
| Install deps | `pip install -r requirements.txt` |
| Install (editable) | `pip install -e .` |
| Lint | `ruff check pyprof/ test/` |
| Type check | `pyright pyprof/` |
| Test | `pytest test/ -v` |

Pre-existing lint/type-check warnings exist in the codebase (unused imports, f-strings without placeholders, operator type issue in `profile_cpu.py`).
