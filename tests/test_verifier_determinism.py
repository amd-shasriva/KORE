"""CPU-only tests for the anti-hack determinism verdict."""

from __future__ import annotations

from kore.env.kore_env import _determinism_stable


def test_stable_when_rerun_matches():
    ok, reason = _determinism_stable(45.0, 45.3, ok2=True, tol_db=10.0)
    assert ok and reason == ""


def test_stable_tolerates_atomic_jitter_within_tolerance():
    # legit atomic-reduction kernels jitter SNR by <~1 dB -> still stable
    ok, _ = _determinism_stable(48.0, 47.2, ok2=True, tol_db=10.0)
    assert ok


def test_unstable_when_second_run_fails_gate():
    ok, reason = _determinism_stable(45.0, None, ok2=False, tol_db=10.0)
    assert not ok and "non-deterministic" in reason


def test_unstable_when_snr_swings_beyond_tolerance():
    # lucky-pass hack: first run 32 dB, re-run collapses to -5 dB
    ok, reason = _determinism_stable(32.0, -5.0, ok2=True, tol_db=10.0)
    assert not ok and "drifted" in reason


def test_missing_snr_but_allclose_true_is_stable():
    # allclose-only drivers report ok2=True with snr2=None -> treated as stable
    ok, _ = _determinism_stable(None, None, ok2=True, tol_db=10.0)
    assert ok
