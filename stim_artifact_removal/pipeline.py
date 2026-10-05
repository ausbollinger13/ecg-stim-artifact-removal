"""One-call cleaning: raw ECG in, cleaned ECG (and diagnostics) out.

    from stim_artifact_removal import clean_ecg
    out = clean_ecg(raw, ts, fs)                      # no stim markers needed
    out = clean_ecg(raw, ts, fs, stim_times=onsets)   # if you have them

Steps, all adapting to the recording's sampling rate ``fs``:

1. decide polarity on an artifact-immune ~1 kHz copy (``detect_inverted``);
2. clean the unfiltered signal at the native rate (despiking, then template
   subtraction), using the markers if they line up with the artifact and
   markers reconstructed from the artifact otherwise;
3. apply the mains notch and baseline high-pass (before cleaning instead
   when ``fs`` < 1 kHz, where mains itself looks spike-like);
4. decimate (anti-aliased) to ``target_fs``.

Cleaning at the native rate matters: QRS curvature shrinks with the square
of the sampling rate while a one-sample spike's does not, so spikes and QRS
separate far better at 5 kHz than at 1 kHz.
"""
from __future__ import annotations

import numpy as np

from .core import (remove_stim_artifact, remove_stim_artifact_no_markers,
                   resolve_curv_thresh)
from .preprocess import (preprocess_ecg, decimate_to, detect_inverted, DEFAULT_TARGET_FS,
                         DEFAULT_LINE_FREQ, DEFAULT_HP_CUTOFF)


def clean_ecg(raw: np.ndarray, ts: np.ndarray | None, fs: float,
              stim_times: np.ndarray | None = None,
              target_fs: float = DEFAULT_TARGET_FS,
              line_freq: float = DEFAULT_LINE_FREQ,
              hp_cutoff: float = DEFAULT_HP_CUTOFF,
              native_rate: bool = True, preprocess: bool = True,
              curv_thresh: float | None = None, phase_locked: bool = False,
              template: bool = True) -> dict:
    """Remove periodic stimulation artifact from one ECG recording.

    Parameters
    ----------
    raw : ECG samples, any units.
    ts : timestamps in seconds (None = evenly spaced from 0 at `fs`).
    fs : sampling rate of `raw`, Hz.
    stim_times : optional stim-onset times (same clock as `ts`). Used only
        if they line up with artifact; otherwise marker-free detection runs.
    target_fs : output rate (decimated to after cleaning).
    native_rate : clean at `fs` (default) rather than after decimating.
    preprocess : False if `raw` is already filtered and upright.

    Returns
    -------
    dict with `ecg` (cleaned), `ecg_preprocessed` (the input to cleaning,
    same rate), `ts`, `fs`, `polarity_flipped`, `mode` (which path ran) and
    `diagnostics` (everything the cleaning functions reported).
    """
    raw = np.asarray(raw, dtype=float)
    ts = np.arange(len(raw)) / fs if ts is None else np.asarray(ts, dtype=float)
    clean_fs = fs if native_rate else min(fs, target_fs)

    inverted = False
    # Clean before the notch / high-pass at >= 1 kHz: filtering a signal that
    # still holds the spikes smears them into stim-locked content (high-pass
    # baseline shifts, notch ringing ~150 ms either side) that no stage can
    # remove afterwards. Below 1 kHz mains looks spike-like, so filter first.
    filter_after = preprocess and clean_fs >= 1000.0
    post = None
    if filter_after:
        if clean_fs < fs:
            raw, ts, fs = decimate_to(raw, ts, fs, clean_fs)
        filt = lambda y: preprocess_ecg(y, clean_fs, target_fs=clean_fs, line_freq=line_freq,
                                        hp_cutoff=hp_cutoff, correct_polarity=False)[0]
        inverted = detect_inverted(filt(raw), clean_fs)
        sig = (raw - np.median(raw)) * (-1.0 if inverted else 1.0)
        post = filt
    elif preprocess:
        sig, factor, clean_fs, inverted = preprocess_ecg(
            raw, fs, target_fs=clean_fs, line_freq=line_freq, hp_cutoff=hp_cutoff)
        ts = ts[::factor][:len(sig)]
    else:
        sig = raw
        if clean_fs < fs:
            sig, ts, clean_fs = decimate_to(sig, ts, fs, clean_fs)

    if curv_thresh is None:
        curv_thresh, _ = resolve_curv_thresh(sig, clean_fs)
    kw = dict(curv_thresh=curv_thresh, phase_locked=phase_locked, template=template)

    mode = "marker-free"
    if stim_times is not None and len(stim_times) >= 3:
        cleaned, diag = remove_stim_artifact(sig, ts, np.asarray(stim_times, float), clean_fs, **kw)
        mode = "markers"
        if diag["window_ms"] is None:
            cleaned, diag = remove_stim_artifact_no_markers(sig, ts, clean_fs, **kw)
            mode = "marker-free (markers did not line up with artifact)"
    else:
        cleaned, diag = remove_stim_artifact_no_markers(sig, ts, clean_fs, **kw)

    if post is not None:
        cleaned, sig = post(cleaned), post(sig)
    out_fs = clean_fs
    out_ts = ts
    if clean_fs > target_fs:
        cleaned, out_ts, out_fs = decimate_to(cleaned, ts, clean_fs, target_fs)
        sig, _, _ = decimate_to(sig, ts, clean_fs, target_fs)
    return {"ecg": cleaned, "ecg_preprocessed": sig, "ts": out_ts, "fs": out_fs,
            "clean_fs": clean_fs, "polarity_flipped": bool(inverted), "mode": mode,
            "diagnostics": diag}
