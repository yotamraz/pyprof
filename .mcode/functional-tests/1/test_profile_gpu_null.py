"""
Functional tests for ProfileGPUNull — the no-op GPU backend.

All tests are target_only (ProfileGPUNull is a new class introduced by the modernization).
"""
import warnings
import pytest

from pyprof.profile_gpu_null import ProfileGPUNull
from pyprof.profile_gpu_base import ProfileGPUBase


class TestProfileGPUNullWarning:
    """ProfileGPUNull() — default behavior emits UserWarning."""

    def test_default_emits_user_warning(self):
        """ProfileGPUNull() with default warn=True emits a UserWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            null = ProfileGPUNull()
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) >= 1

    def test_warning_message_mentions_gpu(self):
        """The UserWarning message refers to GPU being disabled."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ProfileGPUNull()
        msgs = [str(x.message).lower() for x in w if issubclass(x.category, UserWarning)]
        assert any('gpu' in m for m in msgs)

    def test_warn_true_emits_warning(self):
        """Explicit warn=True emits UserWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ProfileGPUNull(warn=True)
        assert any(issubclass(x.category, UserWarning) for x in w)


class TestProfileGPUNullNoWarning:
    """ProfileGPUNull(warn=False) — no UserWarning emitted."""

    def test_warn_false_no_user_warning(self):
        """ProfileGPUNull(warn=False) does not emit a UserWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            null = ProfileGPUNull(warn=False)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_warn_false_is_still_falsy(self):
        """ProfileGPUNull(warn=False) still evaluates to False."""
        null = ProfileGPUNull(warn=False)
        assert not null
        assert bool(null) is False


class TestProfileGPUNullInterface:
    """ProfileGPUNull adheres to ProfileGPUBase interface."""

    def test_is_subclass_of_base(self):
        """ProfileGPUNull inherits from ProfileGPUBase."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            null = ProfileGPUNull()
        assert isinstance(null, ProfileGPUBase)

    def test_get_usage_returns_tuple(self):
        """get_usage() returns a tuple."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            null = ProfileGPUNull()
        result = null.get_usage()
        assert isinstance(result, tuple)

    def test_get_usage_returns_two_floats(self):
        """get_usage() returns exactly two float values."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            null = ProfileGPUNull()
        result = null.get_usage()
        assert len(result) == 2
        util, mem = result
        assert isinstance(util, float)
        assert isinstance(mem, float)

    def test_get_usage_returns_zero_utilization(self):
        """get_usage() returns 0.0 for GPU utilization."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            null = ProfileGPUNull()
        util, mem = null.get_usage()
        assert util == 0.0

    def test_get_usage_returns_zero_memory(self):
        """get_usage() returns 0.0 for GPU memory."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            null = ProfileGPUNull()
        util, mem = null.get_usage()
        assert mem == 0.0

    def test_bool_is_false(self):
        """ProfileGPUNull.__bool__ returns False."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            null = ProfileGPUNull()
        assert bool(null) is False

    def test_get_usage_is_idempotent(self):
        """Multiple calls to get_usage() always return (0.0, 0.0)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            null = ProfileGPUNull()
        for _ in range(5):
            util, mem = null.get_usage()
            assert util == 0.0
            assert mem == 0.0
