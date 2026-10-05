"""ECG preprocessing: decimation, mains notch, baseline high-pass and
polarity detection. ``clean_ecg`` calls these for you; use them directly
only if you are building your own pipeline around the core functions.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, detrend

DEFAULT_TARGET_FS = 1000.0
DEFAULT_HP_CUTOFF = 0.5
DEFAULT_LINE_FREQ = 60.0
DEFAULT_NOTCH_Q = 30.0


def _notch(sig: np.ndarray, fs: float, line_freq: float, q: float) -> np.ndarray:
    if len(sig) < 10 or np.std(sig) == 0:
        return sig
    w0 = line_freq / (fs / 2)
    if not (0 < w0 < 1):
        return sig
    try:
        b, a = iirnotch(w0, q)
        return filtfilt(b, a, sig)
    except Exception:
        return sig


def _highpass(sig: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    if len(sig) < 10 or np.std(sig) == 0:
        return detrend(sig)
    sig = detrend(sig - np.mean(sig))
    norm = cutoff / (fs / 2)
    if not (0 < norm < 1):
        return sig
    try:
        sos = butter(order, norm, btype="high", output="sos")
        return sosfiltfilt(sos, sig)
    except Exception:
        return sig


def preprocess_ecg(raw_sig: np.ndarray, raw_fs: float,
                   target_fs: float = DEFAULT_TARGET_FS,
                   line_freq: float = DEFAULT_LINE_FREQ,
                   hp_cutoff: float = DEFAULT_HP_CUTOFF,
                   correct_polarity: bool = True
                   ) -> tuple[np.ndarray, int, float, bool]:
    """Decimate -> mains notch -> high-pass -> (optional) polarity check.

    ``line_freq`` is the mains frequency (60 Hz in North America, 50 Hz in
    most other countries); it is not auto-detected. Polarity correction
    needs ``neurokit2``; if it is unavailable or fails the signal is left
    upright (``inverted=False``).

    Returns ``(signal, decimation_factor, fs, inverted)``. Apply the same
    factor to the timestamps (``ts[::decimation_factor]``) to keep them
    aligned.
    """
    raw_sig = np.asarray(raw_sig, dtype=np.float64)
    factor = max(1, int(round(raw_fs / target_fs)))
    sig = raw_sig[::factor]
    fs = raw_fs / factor

    sig = _notch(sig, fs, line_freq, DEFAULT_NOTCH_Q)
    sig = _highpass(sig, fs, hp_cutoff)

    inverted = detect_inverted(sig, fs) if correct_polarity else False
    return (-sig if inverted else sig), factor, fs, inverted


def detect_inverted(sig: np.ndarray, fs: float) -> bool:
    """True if the ECG is upside down (NeuroKit2's ``ecg_invert``).

    ``ecg_invert`` is tuned for ~1 kHz ECG and is confused by stimulation
    spikes, so it is run on a 5 ms running-median copy decimated to ~1 kHz.
    Returns False if NeuroKit2 is unavailable or fails.
    """
    try:
        import neurokit2 as nk
        from scipy.ndimage import median_filter
        x = median_filter(np.asarray(sig, dtype=float),
                          size=max(3, int(round(0.005 * fs)) | 1), mode="nearest")
        step = max(1, int(round(fs / 1000.0)))
        _, inverted = nk.ecg_invert(x[::step], sampling_rate=fs / step)
        return bool(inverted)
    except Exception:
        return False


def decimate_to(sig: np.ndarray, ts: np.ndarray, fs: float, target_fs: float
                ) -> tuple[np.ndarray, np.ndarray, float]:
    """Anti-aliased (zero-phase FIR) decimation by the integer factor
    nearest to ``fs / target_fs``. Use after cleaning, when no spikes are
    left to alias. Returns ``(signal, timestamps, new_fs)``.
    """
    from scipy.signal import decimate
    factor = max(1, int(round(fs / target_fs)))
    if factor == 1:
        return np.asarray(sig, dtype=float), np.asarray(ts, dtype=float), float(fs)
    y = decimate(np.asarray(sig, dtype=float), factor, ftype="fir", zero_phase=True)
    return y, np.asarray(ts, dtype=float)[::factor][:len(y)], fs / factor
