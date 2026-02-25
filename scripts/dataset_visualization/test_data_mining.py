"""Unit tests for data_mining.py core functions.

All tests use synthetic data and do not require the real UBFC-Phys dataset.
Run with: conda run -n cctlstm pytest test_data_mining.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import data_mining as dm


# =============================================================================
# Test: _find_optimal_lag_samples
# =============================================================================


class TestFindOptimalLagSamples:
    """Tests for _find_optimal_lag_samples (cross-correlation lag finder)."""

    @staticmethod
    def _make_sine(n: int, freq: float = 0.05) -> np.ndarray:
        t = np.arange(n, dtype=np.float64)
        return np.sin(2 * np.pi * freq * t)

    def test_known_positive_lag(self):
        """signal_a leads signal_b by 10 samples → positive lag."""
        base = self._make_sine(500)
        shift = 10
        # signal_a starts at index 10 of base → it "leads"
        signal_a = base[shift:]
        signal_b = base[: len(signal_a)]
        lag = dm._find_optimal_lag_samples(signal_a, signal_b, max_lag_samples=50)
        assert lag == shift, f"expected lag={shift}, actual lag={lag}"

    def test_known_negative_lag(self):
        """signal_b leads signal_a by 10 samples → negative lag."""
        base = self._make_sine(500)
        shift = 10
        # signal_b starts at index 10 of base → signal_a lags behind
        signal_a = base[: len(base) - shift]
        signal_b = base[shift:]
        lag = dm._find_optimal_lag_samples(signal_a, signal_b, max_lag_samples=50)
        assert lag == -shift, f"expected lag={-shift}, actual lag={lag}"

    def test_zero_lag(self):
        """Identical signals → lag == 0."""
        sig = self._make_sine(300)
        lag = dm._find_optimal_lag_samples(sig, sig.copy(), max_lag_samples=50)
        assert lag == 0, f"expected lag=0, actual lag={lag}"

    def test_lag_bounded_by_max(self):
        """Returned lag must be in [-max_lag, +max_lag]."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(200)
        b = rng.standard_normal(200)
        max_lag = 20
        lag = dm._find_optimal_lag_samples(a, b, max_lag_samples=max_lag)
        assert -max_lag <= lag <= max_lag, (
            f"lag={lag} outside [-{max_lag}, {max_lag}]"
        )

    def test_antisymmetry(self):
        """lag(a, b) == -lag(b, a)."""
        base = self._make_sine(400)
        a = base[5:]
        b = base[: len(a)]
        lag_ab = dm._find_optimal_lag_samples(a, b, max_lag_samples=30)
        lag_ba = dm._find_optimal_lag_samples(b, a, max_lag_samples=30)
        assert lag_ab == -lag_ba, (
            f"antisymmetry violated: lag(a,b)={lag_ab}, lag(b,a)={lag_ba}"
        )


# =============================================================================
# Test: _reconstruct_rppg_bvp_overlap_average
# =============================================================================


class TestReconstructRppgBvpOverlapAverage:
    """Tests for _reconstruct_rppg_bvp_overlap_average."""

    @staticmethod
    def _write_json(path: Path, data) -> None:
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_overlap_average_values(self, tmp_path: Path):
        """Three overlapping windows produce correct averaged values."""
        json_path = tmp_path / "test_rppg.json"
        self._write_json(json_path, [
            {"bvp_signal": [1, 1, 1, 1, 1], "start_frame": 0, "end_frame": 4, "fps": 30},
            {"bvp_signal": [3, 3, 3, 3, 3], "start_frame": 2, "end_frame": 6, "fps": 30},
            {"bvp_signal": [5, 5, 5],        "start_frame": 5, "end_frame": 7, "fps": 30},
        ])

        bvp, fs_int, duration = dm._reconstruct_rppg_bvp_overlap_average(json_path)

        assert bvp is not None
        assert fs_int == 30
        assert len(bvp) == 8, f"expected length 8, actual {len(bvp)}"
        assert duration == pytest.approx(8 / 30)

        np.testing.assert_array_almost_equal(bvp[0:2], [1.0, 1.0])
        np.testing.assert_array_almost_equal(bvp[2:5], [2.0, 2.0, 2.0])
        np.testing.assert_array_almost_equal(bvp[5:7], [4.0, 4.0])
        np.testing.assert_almost_equal(bvp[7], 5.0)

    def test_empty_list(self, tmp_path: Path):
        """Empty JSON list → (None, None, None)."""
        json_path = tmp_path / "empty.json"
        self._write_json(json_path, [])
        result = dm._reconstruct_rppg_bvp_overlap_average(json_path)
        assert result == (None, None, None)

    def test_empty_dict(self, tmp_path: Path):
        """List of empty dicts → (None, None, None)."""
        json_path = tmp_path / "empty_dicts.json"
        self._write_json(json_path, [{}])
        result = dm._reconstruct_rppg_bvp_overlap_average(json_path)
        assert result == (None, None, None)

    def test_nonexistent_file(self, tmp_path: Path):
        """Missing file → (None, None, None)."""
        result = dm._reconstruct_rppg_bvp_overlap_average(
            tmp_path / "does_not_exist.json"
        )
        assert result == (None, None, None)

    def test_inconsistent_fps_skipped(self, tmp_path: Path):
        """Windows with inconsistent fps are skipped; majority fps wins."""
        json_path = tmp_path / "mixed_fps.json"
        self._write_json(json_path, [
            {"bvp_signal": [1, 1, 1], "start_frame": 0, "end_frame": 2, "fps": 30},
            {"bvp_signal": [9, 9, 9], "start_frame": 0, "end_frame": 2, "fps": 60},
        ])
        bvp, fs_int, _ = dm._reconstruct_rppg_bvp_overlap_average(json_path)
        assert bvp is not None
        assert fs_int == 30
        np.testing.assert_array_almost_equal(bvp, [1.0, 1.0, 1.0])

    def test_fps_equals_sampling_rate_key(self, tmp_path: Path):
        """Handles the 'fps=sampling_rate' key used by pyVHR."""
        json_path = tmp_path / "pyvhr.json"
        self._write_json(json_path, [
            {
                "bvp_signal": [2, 2, 2],
                "start_frame": 0,
                "end_frame": 2,
                "fps=sampling_rate": 30,
            },
        ])
        bvp, fs_int, _ = dm._reconstruct_rppg_bvp_overlap_average(json_path)
        assert bvp is not None
        assert fs_int == 30
        np.testing.assert_array_almost_equal(bvp, [2.0, 2.0, 2.0])


# =============================================================================
# Test: calculate_rolling_hr
# =============================================================================


class TestCalculateRollingHR:
    """Tests for calculate_rolling_hr."""

    def test_empty_input(self):
        """Empty BVP signal → two empty arrays."""
        times, hr = dm.calculate_rolling_hr([], fs=64, window_sec=3)
        assert len(times) == 0
        assert len(hr) == 0

    def test_short_signal(self):
        """Signal shorter than one window → empty or does not crash."""
        short_signal = [0.5] * 100  # ~1.6s at 64Hz, less than 3s window
        times, hr = dm.calculate_rolling_hr(short_signal, fs=64, window_sec=3)
        assert isinstance(times, np.ndarray)
        assert isinstance(hr, np.ndarray)

    def test_synthetic_sine_hr_range(self):
        """Synthetic 72bpm sine wave → HR values in plausible range [40, 140]."""
        fs = 64
        duration_sec = 30
        target_hr_bpm = 72
        freq_hz = target_hr_bpm / 60.0

        t = np.arange(duration_sec * fs) / fs
        bvp = np.sin(2 * np.pi * freq_hz * t).tolist()

        times, hr = dm.calculate_rolling_hr(bvp, fs=fs, window_sec=3, step_sec=0.5)

        assert len(times) > 0, "Expected non-empty time array for 30s signal"
        assert len(hr) == len(times)

        valid_hr = hr[~np.isnan(hr)]
        if len(valid_hr) > 0:
            assert np.all(valid_hr > 40), (
                f"HR values below 40: min={np.nanmin(valid_hr):.1f}"
            )
            assert np.all(valid_hr < 140), (
                f"HR values above 140: max={np.nanmax(valid_hr):.1f}"
            )

    def test_output_types(self):
        """Return types are numpy arrays regardless of input."""
        fs = 64
        bvp = (np.sin(2 * np.pi * 1.0 * np.arange(20 * fs) / fs)).tolist()
        times, hr = dm.calculate_rolling_hr(bvp, fs=fs, window_sec=3)
        assert isinstance(times, np.ndarray)
        assert isinstance(hr, np.ndarray)

    def test_time_monotonic(self):
        """Time points are monotonically non-decreasing."""
        fs = 64
        bvp = (np.sin(2 * np.pi * 1.2 * np.arange(20 * fs) / fs)).tolist()
        times, _ = dm.calculate_rolling_hr(bvp, fs=fs, window_sec=3)
        if len(times) > 1:
            diffs = np.diff(times)
            assert np.all(diffs >= 0), "Time points are not monotonically non-decreasing"
