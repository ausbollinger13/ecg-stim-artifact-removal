import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from stim_artifact_removal import clean_ecg, ecg_hrv


def stim_locked_hf_ptp(x, ts, stim, fs):
    """Peak-to-peak of the stim-locked average of the >100 Hz component."""
    h = sosfiltfilt(butter(4, 100.0, "high", fs=fs, output="sos"), x)
    idx = np.searchsorted(ts, stim)
    n = int(0.5 * fs)
    idx = idx[idx + n < len(h)]
    return float(np.ptp(np.mean([h[i:i + n] for i in idx], axis=0)))


@pytest.mark.parametrize("use_markers", [True, False])
def test_spikes_removed_to_noise_floor(recording, use_markers):
    r = recording
    out = clean_ecg(r["spikes_only"], r["ts"], r["fs"],
                    stim_times=r["stim"] if use_markers else None)
    assert out["diagnostics"]["window_ms"] is not None
    assert out["mode"] == ("markers" if use_markers else "marker-free")
    ref = clean_ecg(r["clean"], r["ts"], r["fs"])["ecg"]
    after = stim_locked_hf_ptp(out["ecg"], out["ts"], r["stim"], out["fs"])
    floor = stim_locked_hf_ptp(ref, out["ts"], r["stim"], out["fs"])
    assert after < 2 * floor


@pytest.mark.parametrize("use_markers", [True, False])
def test_spikes_with_tails_removed(recording, use_markers):
    # Smooth after-effects smaller than the beat-to-beat background are left
    # in place by design (see docs/method.md), so require a >98% reduction
    # rather than the noise floor.
    r = recording
    out = clean_ecg(r["dirty"], r["ts"], r["fs"],
                    stim_times=r["stim"] if use_markers else None)
    before = stim_locked_hf_ptp(out["ecg_preprocessed"], out["ts"], r["stim"], out["fs"])
    after = stim_locked_hf_ptp(out["ecg"], out["ts"], r["stim"], out["fs"])
    assert after < 0.02 * before
    assert after < 0.01 * np.percentile(np.abs(out["ecg"]), 99.9)


def test_clean_recording_untouched(recording):
    r = recording
    out = clean_ecg(r["clean"], r["ts"], r["fs"])
    assert out["diagnostics"]["window_ms"] is None
    np.testing.assert_array_equal(out["ecg"], out["ecg_preprocessed"])


def test_markers_without_artifact_fall_back(recording):
    r = recording
    out = clean_ecg(r["clean"], r["ts"], r["fs"], stim_times=r["stim"])
    assert out["mode"].startswith("marker-free")
    np.testing.assert_array_equal(out["ecg"], out["ecg_preprocessed"])


def test_output_rate_and_hrv(recording):
    r = recording
    out = clean_ecg(r["dirty"], r["ts"], r["fs"])
    assert out["fs"] == 1000.0 and out["clean_fs"] == r["fs"]
    assert len(out["ecg"]) == len(out["ts"])
    truth = ecg_hrv(clean_ecg(r["clean"], r["ts"], r["fs"])["ecg"], out["fs"])
    got = ecg_hrv(out["ecg"], out["fs"])
    assert got["n_peaks"] == truth["n_peaks"]
    assert abs(got["MeanNN"] - truth["MeanNN"]) < 1.0
    assert abs(got["RMSSD"] - truth["RMSSD"]) < 0.05 * truth["RMSSD"] + 1.0


def test_inverted_input_is_flipped(recording):
    r = recording
    up = clean_ecg(r["dirty"], r["ts"], r["fs"])
    down = clean_ecg(-r["dirty"], r["ts"], r["fs"])
    assert up["polarity_flipped"] != down["polarity_flipped"]
    np.testing.assert_allclose(up["ecg"], down["ecg"], atol=1e-6 * np.ptp(up["ecg"]))
