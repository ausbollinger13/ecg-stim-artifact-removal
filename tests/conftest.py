import numpy as np
import pytest


def synthetic_ecg(fs=5000.0, duration=60.0, seed=0):
    import neurokit2 as nk
    ecg = nk.ecg_simulate(duration=duration, sampling_rate=int(fs), heart_rate=70,
                          heart_rate_std=3, random_state=seed) * 1000.0
    rng = np.random.default_rng(seed)
    return ecg + rng.normal(0, 5.0, len(ecg))


def add_stim_artifact(ecg, fs, period=1.0, first=0.73, n_pulses=20, pulse_hz=80.0,
                      amp=8000.0, tail_ms=3.0, seed=1):
    """Trains of biphasic one-sample spikes with a short exponential tail,
    one train per `period` seconds. Returns (signal, train_onset_times)."""
    rng = np.random.default_rng(seed)
    x = ecg.copy()
    onsets = np.arange(first, len(x) / fs - period, period)
    n_tail = int(tail_ms * fs / 1000)
    tail = np.exp(-np.arange(n_tail) / (0.5 * n_tail)) if n_tail else np.zeros(0)
    for t0 in onsets:
        for k in range(n_pulses):
            i = int(round((t0 + k / pulse_hz) * fs))
            a = amp * (1 + 0.05 * rng.standard_normal())
            x[i] += a
            x[i + 1] -= a
            x[i + 2: i + 2 + len(tail)] += 0.02 * a * tail
    return x, onsets


@pytest.fixture(scope="session")
def recording():
    fs = 5000.0
    clean = synthetic_ecg(fs)
    dirty, onsets = add_stim_artifact(clean, fs)
    spikes_only, _ = add_stim_artifact(clean, fs, tail_ms=0)
    ts = np.arange(len(clean)) / fs
    return {"fs": fs, "ts": ts, "clean": clean, "dirty": dirty, "spikes_only": spikes_only,
            "stim": onsets}
