"""Core algorithms: find periodic stimulation artifact in an ECG and remove it.

Two stages, both restricted to a per-cycle window found from the data:

1. **Despike.** A QRS changes smoothly from sample to sample; a stimulation
   spike does not. Samples whose local curvature
   ``x[i] - (x[i-1] + x[i+1]) / 2`` exceeds a threshold are replaced by
   linear interpolation.
2. **Template subtraction.** Smooth, stimulation-locked after-effects that
   despiking cannot catch (amplifier recovery, spike tails) are estimated as
   the per-offset median across all stimulation cycles and subtracted. A
   heartbeat is not locked to the stimulation, so it does not enter the
   median.

The timing of the artifact is never assumed. With stim markers, the window
is calibrated from the latency at which curvature events recur after each
marker; without markers, the stimulation period and phase are recovered from
the curvature events themselves. If no periodic artifact is found the
signal is returned unchanged.

See ``docs/method.md`` for a full description and the validation behind
each design choice.
"""
from __future__ import annotations

import numpy as np

DEFAULT_CURV_THRESH = 10000.0
DEFAULT_CURV_THRESH_REF_FS = 1000.0
DEFAULT_PAD = 1
DEFAULT_PAD_MS = 1.0
DEFAULT_TEMPLATE_HP_HZ = 10.0
DEFAULT_TEMPLATE_K_SE = 5.0
DEFAULT_TEMPLATE_MAX_SHIFT_MS = 5.0
DEFAULT_TEMPLATE_TAIL_MS = 100.0
DEFAULT_TEMPLATE_ALIGN_HP_HZ = 100.0
DEFAULT_BIN_MS = 20.0
DEFAULT_ELEVATED_RATIO = 4.0
DEFAULT_MAX_GAP_MS = 80.0
DEFAULT_CALIB_PAD_MS = 15.0
DEFAULT_CALIB_PRE_FRAC = 0.05
DEFAULT_QRS_GUARD = 2.0
DEFAULT_QRS_SENSITIVITY = 8.0
DEFAULT_CALIB_BASELINE_PCT = 25.0
DEFAULT_MIN_PERIOD_S = 0.2
DEFAULT_MAX_PERIOD_S = 10.0
DEFAULT_MIN_EVENTS_FOR_PERIODICITY = 20
DEFAULT_AUTOCORR_BIN_MS = 20.0
DEFAULT_AUTOCORR_ELEVATED_RATIO = 4.0
DEFAULT_RAYLEIGH_ALPHA = 0.02
DEFAULT_RAYLEIGH_OVERSAMPLE = 3.0
DEFAULT_RAYLEIGH_MIN_CYCLES = 30
DEFAULT_PHASE_LOCK_PERCENTILE = 50.0
DEFAULT_PHASE_LOCK_MIN_CYCLES = 20


def default_pad_for_fs(fs: float) -> int:
    """Dilation (in samples) around each flagged sample: ``DEFAULT_PAD_MS`` at
    this sampling rate, so a spike's decaying tail is covered at any rate."""
    return max(1, int(round(DEFAULT_PAD_MS * fs / 1000.0)))


def default_curv_thresh_for_fs(fs: float) -> float:
    """Base curvature threshold for sampling rate ``fs``.

    A smooth waveform's curvature scales as ``1/fs**2``, so the threshold is
    raised below the 1 kHz reference. It is not lowered above it: noise and
    single-sample spikes have roughly rate-independent curvature, and a
    lowered threshold would fall into the noise floor.
    """
    return DEFAULT_CURV_THRESH * max(1.0, (DEFAULT_CURV_THRESH_REF_FS / fs) ** 2)


def estimate_qrs_curvature(sig: np.ndarray, fs: float) -> float | None:
    """Typical QRS curvature of this recording, or None if beats can't be found.

    Beats are located on a 5 ms running-median copy (which deletes 1-3 sample
    stimulation spikes but keeps the ~100 ms QRS), band-passed 0.5-40 Hz; the
    curvature is then measured on the original signal within +-2 ms of each
    R tip. The 25th percentile across beats is returned, so a spike that lands
    on a tip (which can only raise that beat's value) does not inflate it.
    """
    try:
        import neurokit2 as nk
        from scipy.ndimage import median_filter
        from scipy.signal import butter, sosfiltfilt
    except ImportError:
        return None
    x = np.asarray(sig, dtype=float)
    if len(x) < 10 * fs or not np.all(np.isfinite(x)):
        return None
    try:
        mf = median_filter(x, size=max(3, int(round(0.005 * fs)) | 1), mode="nearest")
        lp = sosfiltfilt(butter(4, [0.5, min(40.0, 0.45 * fs)], "bandpass", fs=fs, output="sos"), mf)
        _, info = nk.ecg_peaks(lp, sampling_rate=fs)
        peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception:
        return None
    duration_s = len(x) / fs
    if not (0.5 * duration_s <= len(peaks) <= 3.5 * duration_s):   # 30-210 bpm
        return None
    c = np.abs(_curvature(x))
    h = max(1, int(round(0.01 * fs)))
    w = max(1, int(round(0.002 * fs)))
    vals = []
    for p in peaks:
        if p - h - 2 * w < 1 or p + h + 2 * w + 1 > len(x) - 1:
            continue
        tip_mf = p - h + int(np.argmax(np.abs(mf[p - h: p + h + 1])))
        tip = tip_mf - w + int(np.argmax(np.abs(x[tip_mf - w: tip_mf + w + 1])))
        vals.append(c[tip - w: tip + w + 1].max())
    if len(vals) < 10:
        return None
    return float(np.percentile(vals, 25))


def resolve_curv_thresh(sig: np.ndarray, fs: float,
                        guard: float = DEFAULT_QRS_GUARD,
                        sensitivity: float = DEFAULT_QRS_SENSITIVITY) -> tuple[float, dict]:
    """Curvature threshold used whenever ``curv_thresh`` is left as None.

    Starts from ``default_curv_thresh_for_fs(fs)`` and clamps it to
    ``[guard, sensitivity]`` x the recording's own QRS curvature. The lower
    bound keeps R peaks from being treated as spikes on high-gain recordings;
    the upper bound brings the threshold into range for recordings in other
    units (e.g. millivolts). If beats can't be found the default is used.

    Returns ``(curv_thresh, info)`` with info keys ``curv_thresh``,
    ``default``, ``qrs_curvature`` and ``rule``.
    """
    default = default_curv_thresh_for_fs(fs)
    qrs = estimate_qrs_curvature(sig, fs)
    info = {"default": round(default, 1),
            "qrs_curvature": None if qrs is None else round(qrs, 3)}
    if qrs is None or qrs <= 0:
        thresh, rule = default, "default (heartbeats not found)"
    elif default < guard * qrs:
        thresh, rule = guard * qrs, f"raised to {guard:g}x QRS curvature (protects R peaks)"
    elif default > sensitivity * qrs:
        thresh, rule = sensitivity * qrs, f"lowered to {sensitivity:g}x QRS curvature (signal scale)"
    else:
        thresh, rule = default, "default (within the QRS-relative bounds)"
    info.update(curv_thresh=round(float(thresh), 3), rule=rule)
    return float(thresh), info


def _curvature(seg: np.ndarray) -> np.ndarray:
    """Deviation of each sample from the straight line through its neighbours."""
    c = np.zeros_like(seg)
    c[1:-1] = seg[1:-1] - 0.5 * (seg[:-2] + seg[2:])
    return c


def _curvature_events(seg: np.ndarray, curv_thresh: float, pad: int):
    """Mask of samples to remove (flagged samples dilated by ``pad``) and the
    flagged indices themselves."""
    n = len(seg)
    c = _curvature(seg)
    flagged = np.where(np.abs(c) > curv_thresh)[0]
    bad = np.zeros(n, dtype=bool)
    for i in flagged:
        bad[max(0, i - pad): min(n, i + pad + 1)] = True
    return bad, flagged


def calibrate_artifact_window(sig: np.ndarray, ts: np.ndarray, stim_times: np.ndarray,
                              fs: float, search_frac: float = 0.95,
                              curv_thresh: float | None = None,
                              bin_ms: float = DEFAULT_BIN_MS,
                              elevated_ratio: float = DEFAULT_ELEVATED_RATIO,
                              max_gap_ms: float = DEFAULT_MAX_GAP_MS,
                              pad_ms: float = DEFAULT_CALIB_PAD_MS,
                              pre_frac: float = DEFAULT_CALIB_PRE_FRAC,
                              baseline_pct: float = DEFAULT_CALIB_BASELINE_PCT
                              ) -> tuple[tuple[float, float] | None, dict]:
    """Find the window (ms relative to each marker) that holds the artifact.

    Curvature events are collected over most of each stimulation period
    (from ``pre_frac`` of the period before each marker, since a software
    marker can lag the hardware pulses) and their latencies histogrammed.
    Bins well above the ``baseline_pct`` percentile of bin counts form the
    artifact cluster; nearby elevated bins are merged across gaps up to
    ``max_gap_ms`` (fast pulse trains alias into a comb of bins).

    Returns ``((start_ms, end_ms), diagnostics)``, or ``(None, diagnostics)``
    when no stimulation-locked cluster exists.
    """
    if curv_thresh is None:
        curv_thresh, _ = resolve_curv_thresh(sig, fs)
    stim_times = np.asarray(stim_times, dtype=float)
    if len(stim_times) < 3:
        return None, {"reason": "too few stim markers to calibrate", "baseline": 0.0}
    period_ms = float(np.median(np.diff(stim_times))) * 1000.0
    search_ms = (-period_ms * pre_frac, period_ms * (search_frac - pre_frac))

    w0 = int(np.floor(search_ms[0] * fs / 1000))
    w1 = int(search_ms[1] * fs / 1000)
    latencies = []
    for st in stim_times:
        idx = int(np.searchsorted(ts, st))
        if idx + w1 >= len(sig) or idx + w0 < 0:
            continue
        seg = sig[idx + w0: idx + w1]
        _, flagged = _curvature_events(seg, curv_thresh, pad=0)
        latencies += [(i + w0) * 1000.0 / fs for i in flagged]
    latencies = np.asarray(latencies)

    bins = np.arange(search_ms[0], search_ms[1] + bin_ms, bin_ms)
    if len(bins) < 2:
        return None, {"reason": "search window too short to bin", "baseline": 0.0}
    counts, edges = np.histogram(latencies, bins=bins)
    baseline = float(np.percentile(counts, baseline_pct))
    elevated = counts > max(baseline * elevated_ratio, baseline + 5)
    if not elevated.any():
        return None, {"reason": "no elevated cluster found -- likely no "
                                "stim-locked artifact present", "baseline": baseline}

    idx = np.where(elevated)[0]
    max_gap_bins = int(np.ceil(max_gap_ms / bin_ms))
    runs = np.split(idx, np.where(np.diff(idx) > max_gap_bins)[0] + 1)
    best = max(runs, key=len)
    w_start = max(search_ms[0], float(edges[best[0]]) - pad_ms)
    w_end = float(edges[best[-1] + 1]) + pad_ms
    peak_ratio = float(counts[best].max() / max(baseline, 1))
    return (w_start, w_end), {"baseline": baseline, "peak_ratio": round(peak_ratio, 1),
                              "n_bins_in_cluster": len(best), "n_events": len(latencies)}


def despike_window(seg: np.ndarray, curv_thresh: float = DEFAULT_CURV_THRESH,
                   pad: int = DEFAULT_PAD, extra_flagged=None) -> tuple[np.ndarray, int]:
    """Interpolate over curvature-flagged samples in one window.

    ``curv_thresh`` is used as given (no sampling-rate scaling).
    ``extra_flagged`` are additional offsets to remove regardless of their
    curvature (used by the ``phase_locked`` option). Returns
    ``(cleaned_segment, n_flagged)``.
    """
    seg = np.asarray(seg, dtype=float).copy()
    bad, flagged = _curvature_events(seg, curv_thresh, pad)
    flagged = list(flagged)
    if extra_flagged is not None and len(extra_flagged) > 0:
        n = len(seg)
        for i in extra_flagged:
            i = int(i)
            if i not in flagged:
                flagged.append(i)
            bad[max(0, i - pad): min(n, i + pad + 1)] = True
    if not bad.any():
        return seg, 0
    good_idx = np.where(~bad)[0]
    if len(good_idx) < 4:
        return seg, len(flagged)
    bad_idx = np.where(bad)[0]
    seg[bad_idx] = np.interp(bad_idx, good_idx, seg[good_idx])
    return seg, len(flagged)


def _phase_locked_offsets(sig: np.ndarray, ts: np.ndarray, stim_times: np.ndarray,
                          fs: float, window_ms: tuple[float, float],
                          elevated_ratio: float = DEFAULT_ELEVATED_RATIO,
                          percentile: float = DEFAULT_PHASE_LOCK_PERCENTILE,
                          min_cycles: int = DEFAULT_PHASE_LOCK_MIN_CYCLES
                          ) -> tuple[np.ndarray, dict]:
    """Window offsets whose curvature is consistently elevated across cycles.

    For weak artifact whose single-cycle curvature does not clear the
    threshold: the ``percentile`` (default median) of curvature at each
    offset across all cycles highlights offsets that are stimulation-locked,
    while QRS content (not locked to stimulation) averages out.
    """
    w0 = int(window_ms[0] * fs / 1000)
    w1 = int(window_ms[1] * fs / 1000)
    rows = []
    for st in stim_times:
        idx = int(np.searchsorted(ts, st))
        if idx + w1 >= len(sig) or idx + w0 < 0:
            continue
        rows.append(np.abs(_curvature(sig[idx + w0: idx + w1])))
    if len(rows) < min_cycles:
        return np.array([], dtype=int), {
            "reason": f"too few cycles ({len(rows)} < {min_cycles}) for phase-locked detection"}

    mat = np.array(rows)
    profile = np.percentile(mat, percentile, axis=0)
    baseline = float(np.median(profile))
    thresh = max(baseline * elevated_ratio, baseline + 5)
    elevated = profile > thresh
    if not elevated.any():
        return np.array([], dtype=int), {
            "reason": "no phase-locked offsets found beyond the per-cycle check",
            "baseline": baseline, "n_cycles_used": len(rows)}

    offsets = np.where(elevated)[0]
    return offsets, {"baseline": round(baseline, 1), "n_locked_offsets": int(len(offsets)),
                     "peak_ratio": round(float(profile.max() / max(baseline, 1)), 1),
                     "n_cycles_used": len(rows)}


def subtract_stim_template(sig: np.ndarray, ts: np.ndarray, stim_times: np.ndarray,
                           fs: float, hp_hz: float = DEFAULT_TEMPLATE_HP_HZ,
                           k_se: float = DEFAULT_TEMPLATE_K_SE,
                           max_shift_ms: float = DEFAULT_TEMPLATE_MAX_SHIFT_MS,
                           n_iter: int = 2, window_ms: tuple[float, float] | None = None,
                           tail_ms: float = DEFAULT_TEMPLATE_TAIL_MS,
                           align_hp_hz: float | None = DEFAULT_TEMPLATE_ALIGN_HP_HZ
                           ) -> tuple[np.ndarray, dict]:
    """Subtract the stimulation-locked template of what despiking left.

    The template is the per-offset median, across all cycles, of the
    signal's >``hp_hz`` component over one period; offsets whose median is
    not significantly non-zero (``k_se`` robust standard errors) are zeroed.
    Each cycle's alignment is refined within +-``max_shift_ms`` using the
    sharp (>``align_hp_hz``) content, absorbing marker jitter. With
    ``window_ms`` the template is confined to that window +-``tail_ms``.
    Needs at least 20 cycles; otherwise the input is returned unchanged.
    """
    from scipy.signal import butter, sosfiltfilt
    y = np.asarray(sig, dtype=float)
    stim_times = np.asarray(stim_times, dtype=float)
    if len(stim_times) < 20:
        return y.copy(), {"reason": "too few cycles for a template"}
    h = sosfiltfilt(butter(2, hp_hz, "high", fs=fs, output="sos"), y)
    ha = h if not align_hp_hz or align_hp_hz >= 0.45 * fs else \
        sosfiltfilt(butter(2, align_hp_hz, "high", fs=fs, output="sos"), y)
    period = float(np.median(np.diff(stim_times)))
    length = int(0.98 * period * fs)
    w0 = int(np.floor(-DEFAULT_CALIB_PRE_FRAC * period * fs))
    ms = max(1, int(round(max_shift_ms * fs / 1000.0)))
    starts = np.searchsorted(ts, stim_times) + w0
    starts = starts[(starts - ms >= 0) & (starts + length + ms < len(y))]
    if len(starts) < 20:
        return y.copy(), {"reason": "too few complete cycles for a template"}
    shifts = np.zeros(len(starts), dtype=int)
    template = np.zeros(length)
    for it in range(n_iter + 1):
        m = np.array([h[a + d: a + d + length] for a, d in zip(starts, shifts)])
        med = np.median(m, axis=0)
        se = 1.253 * 1.4826 * np.median(np.abs(m - med), axis=0) / np.sqrt(len(m))
        template = np.where(np.abs(med) > k_se * np.maximum(se, 1e-12), med, 0.0)
        if window_ms is not None:
            off_ms = (np.arange(length) + w0) * 1000.0 / fs
            template[(off_ms < window_ms[0] - tail_ms) | (off_ms > window_ms[1] + tail_ms)] = 0.0
        if it == n_iter or not template.any():
            break
        ta = np.median(np.array([ha[a + d: a + d + length] for a, d in zip(starts, shifts)]), axis=0)
        ta[template == 0] = 0.0
        for j, a in enumerate(starts):
            score = [np.dot(ha[a + d: a + d + length], ta) for d in range(-ms, ms + 1)]
            shifts[j] = int(np.argmax(score)) - ms
    out = y.copy()
    for a, d in zip(starts, shifts):
        out[a + d: a + d + length] -= template
    return out, {"n_cycles": int(len(starts)),
                 "frac_offsets": round(float((template != 0).mean()), 3),
                 "template_ptp": round(float(np.ptp(template)), 3),
                 "shift_spread_ms": round(float(np.std(shifts) / fs * 1000.0), 3)}


def remove_stim_artifact(sig: np.ndarray, ts: np.ndarray, stim_times: np.ndarray,
                         fs: float, window_ms: tuple[float, float] | None = None,
                         curv_thresh: float | None = None, pad: int | None = None,
                         calibration_kwargs: dict | None = None,
                         phase_locked: bool = False,
                         phase_locked_kwargs: dict | None = None,
                         template: bool = True
                         ) -> tuple[np.ndarray, dict]:
    """Remove stimulation artifact using known stim-onset times.

    Parameters
    ----------
    sig : ECG samples (any units), already at the rate you want to clean at.
    ts : timestamps of ``sig`` in seconds, same clock as ``stim_times``.
    stim_times : stimulation onset times (one per stimulation cycle).
    fs : sampling rate of ``sig`` in Hz.
    window_ms : artifact window (start, end) in ms relative to each marker.
        None (default) calibrates it from the data.
    curv_thresh : curvature threshold. None (default) uses
        ``resolve_curv_thresh``.
    pad : samples removed either side of each flagged sample. None uses
        ``default_pad_for_fs`` (1 ms).
    phase_locked : also remove offsets that are consistently elevated across
        cycles even if no single cycle clears the threshold (weak artifact).
    template : subtract the stimulation-locked template after despiking.

    Returns
    -------
    ``(cleaned, diagnostics)``. If no artifact is found, ``window_ms`` in the
    diagnostics is None and the signal is returned unchanged.
    """
    thresh_info = {"curv_thresh": curv_thresh, "rule": "explicit"}
    if curv_thresh is None:
        curv_thresh, thresh_info = resolve_curv_thresh(sig, fs)
    if pad is None:
        pad = default_pad_for_fs(fs)
    calibration = None
    if window_ms is None:
        window_ms, calibration = calibrate_artifact_window(
            sig, ts, stim_times, fs, curv_thresh=curv_thresh,
            **(calibration_kwargs or {}))
        if window_ms is None:
            return np.asarray(sig, dtype=float).copy(), {
                "n_cycles": 0, "n_spikes": 0, "window_ms": None,
                "calibration": calibration, "threshold": thresh_info}

    locked_offsets = None
    phase_locked_diag = None
    if phase_locked:
        locked_offsets, phase_locked_diag = _phase_locked_offsets(
            sig, ts, stim_times, fs, window_ms, **(phase_locked_kwargs or {}))

    w0 = int(window_ms[0] * fs / 1000)
    w1 = int(window_ms[1] * fs / 1000)
    cleaned = np.asarray(sig, dtype=float).copy()
    n_spikes = 0
    n_cycles = 0
    for st in stim_times:
        idx = int(np.searchsorted(ts, st))
        if idx + w1 >= len(sig) or idx + w0 < 0:
            continue
        n_cycles += 1
        seg_clean, n = despike_window(sig[idx + w0: idx + w1], curv_thresh, pad,
                                      extra_flagged=locked_offsets)
        cleaned[idx + w0: idx + w1] = seg_clean
        n_spikes += n
    diag = {"n_cycles": n_cycles, "n_spikes": n_spikes,
            "window_ms": window_ms, "calibration": calibration, "threshold": thresh_info}
    if template:
        cleaned, diag["template"] = subtract_stim_template(cleaned, ts, stim_times, fs,
                                                           window_ms=window_ms)
    if phase_locked:
        diag["phase_locked"] = phase_locked_diag
    return cleaned, diag


def _estimate_period(flag_ts: np.ndarray, duration_s: float,
                     bin_ms: float = DEFAULT_AUTOCORR_BIN_MS,
                     min_period_s: float = DEFAULT_MIN_PERIOD_S,
                     max_period_s: float = DEFAULT_MAX_PERIOD_S,
                     elevated_ratio: float = DEFAULT_AUTOCORR_ELEVATED_RATIO,
                     max_gap_ms: float = DEFAULT_MAX_GAP_MS
                     ) -> tuple[float | None, dict]:
    """Dominant recurrence period of event times (s from recording start),
    from the autocorrelation of a binned event train."""
    n_bins = int(np.ceil(duration_s * 1000.0 / bin_ms)) + 1
    max_lag_bins = int(max_period_s * 1000.0 / bin_ms)
    min_lag_bins = max(1, int(min_period_s * 1000.0 / bin_ms))
    if n_bins <= max_lag_bins or max_lag_bins <= min_lag_bins:
        return None, {"reason": "recording too short for max_period_s search window"}

    indicator = np.zeros(n_bins)
    idx = np.clip((flag_ts * 1000.0 / bin_ms).astype(int), 0, n_bins - 1)
    indicator[idx] = 1.0

    ac_full = np.correlate(indicator, indicator, mode="full")
    mid = len(ac_full) // 2
    ac = ac_full[mid: mid + max_lag_bins + 1]

    search = ac[min_lag_bins:]
    baseline = float(np.median(search))
    thresh = max(baseline * elevated_ratio, baseline + 5)
    elevated = search > thresh
    if not elevated.any():
        return None, {"reason": "no periodic recurrence found in flagged events",
                      "baseline": baseline}

    # Merge elevated lags into clusters (a fast pulse train aliases one
    # resonance into a comb of bins); each cluster's peak is one candidate.
    idx = np.where(elevated)[0]
    max_gap_bins = int(np.ceil(max_gap_ms / bin_ms))
    runs = np.split(idx, np.where(np.diff(idx) > max_gap_bins)[0] + 1)
    candidates = np.array(sorted(int(r[np.argmax(search[r])]) for r in runs))

    n = len(search)

    def _harmonic_support(i):
        # The true burst period also shows elevated autocorrelation at its
        # 2nd and 3rd multiples; an intra-burst pulse interval does not.
        abs_i = i + min_lag_bins
        n_checked = n_hit = 0
        for k in (2, 3):
            j = abs_i * k - min_lag_bins
            if j >= n:
                break
            n_checked += 1
            lo, hi = max(0, j - max_gap_bins), min(n, j + max_gap_bins + 1)
            if elevated[lo:hi].any():
                n_hit += 1
        return n_checked, n_hit

    # Accept the smallest candidate whose harmonics were checked and all
    # present; otherwise fall back to the strongest candidate.
    accepted = None
    for i in candidates.tolist():
        n_checked, n_hit = _harmonic_support(i)
        if n_checked > 0 and n_hit == n_checked:
            accepted = i
            break
    if accepted is None:
        accepted = int(candidates[np.argmax(search[candidates])])

    peak_i = accepted + min_lag_bins
    peak_val = float(ac[peak_i])
    period_s = peak_i * bin_ms / 1000.0
    return period_s, {"baseline": baseline,
                      "peak_ratio": round(peak_val / max(baseline, 1), 1),
                      "period_s": period_s}


def _dominant_phase(flag_ts: np.ndarray, t_ref: float, period_s: float,
                    bin_ms: float = DEFAULT_AUTOCORR_BIN_MS,
                    elevated_ratio: float = DEFAULT_AUTOCORR_ELEVATED_RATIO,
                    max_gap_ms: float = DEFAULT_MAX_GAP_MS) -> float | None:
    """Phase (s into the cycle, relative to ``t_ref``) where events cluster,
    found circularly so a cluster spanning the cycle boundary stays whole."""
    period_ms = period_s * 1000.0
    phases_ms = ((flag_ts - t_ref) * 1000.0) % period_ms
    bins = np.arange(0.0, period_ms + bin_ms, bin_ms)
    n_bins = len(bins) - 1
    if n_bins < 2:
        return None
    counts, edges = np.histogram(phases_ms, bins=bins)
    baseline = float(np.median(counts))
    elevated = counts > max(baseline * elevated_ratio, baseline + 5)
    if not elevated.any():
        return None

    idx = np.where(elevated)[0]
    max_gap_bins = int(np.ceil(max_gap_ms / bin_ms))
    idx_ext = np.sort(np.concatenate([idx, idx + n_bins]))
    splits = np.where(np.diff(idx_ext) > max_gap_bins)[0] + 1
    runs = [np.unique(r % n_bins) for r in np.split(idx_ext, splits)]
    best = max(runs, key=len)
    centers_ms = edges[best] + bin_ms / 2.0
    angles = centers_ms / period_ms * 2 * np.pi
    mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    center_ms = (mean_angle / (2 * np.pi)) % 1.0 * period_ms
    return center_ms / 1000.0


def _artifact_arc_center(flag_ts: np.ndarray, t_ref: float, period_s: float,
                         bin_ms: float = DEFAULT_BIN_MS,
                         elevated_ratio: float = DEFAULT_ELEVATED_RATIO,
                         baseline_pct: float = DEFAULT_CALIB_BASELINE_PCT) -> float | None:
    """Centre of the part of the cycle that holds artifact, taken as the side
    opposite the longest quiet stretch. Unlike a density-weighted mean, this
    is not pulled off-centre by uneven event density inside a long train."""
    period_ms = period_s * 1000.0
    n_bins = int(period_ms // bin_ms)
    if n_bins < 4:
        return None
    phases = ((flag_ts - t_ref) * 1000.0) % period_ms
    counts, edges = np.histogram(phases, bins=np.linspace(0.0, period_ms, n_bins + 1))
    baseline = float(np.percentile(counts, baseline_pct))
    elevated = counts > max(baseline * elevated_ratio, baseline + 5)
    if not elevated.any() or elevated.all():
        return None
    quiet = np.concatenate([~elevated, ~elevated])
    best_len, best_end, run = 0, 0, 0
    for i, q in enumerate(quiet):
        run = run + 1 if q else 0
        if run > best_len:
            best_len, best_end = min(run, n_bins), i
    gap_center_bin = best_end - (best_len - 1) / 2.0
    width = period_ms / n_bins
    gap_center_ms = ((gap_center_bin + 0.5) * width) % period_ms
    return ((gap_center_ms + period_ms / 2.0) % period_ms) / 1000.0


def _rayleigh_period_phase(flag_ts: np.ndarray, duration_s: float,
                           min_period_s: float = DEFAULT_MIN_PERIOD_S,
                           max_period_s: float = DEFAULT_MAX_PERIOD_S,
                           oversample: float = DEFAULT_RAYLEIGH_OVERSAMPLE,
                           alpha: float = DEFAULT_RAYLEIGH_ALPHA,
                           min_cycles: float = DEFAULT_RAYLEIGH_MIN_CYCLES
                           ) -> tuple[float | None, float | None, dict]:
    """Fallback period/phase estimate for sparse events (Rayleigh test).

    Folds all event times onto each candidate period and measures phase
    concentration, pooling evidence from every cycle. Candidate periods are
    limited to those observed at least ``min_cycles`` times; significance
    uses a Bonferroni correction over ``bandwidth x duration`` independent
    frequencies. Returns ``(period_s, phase_s, diagnostics)`` or
    ``(None, None, diagnostics)``.
    """
    n = len(flag_ts)
    t = flag_ts - flag_ts[0]
    f_min = max(1.0 / max_period_s, min_cycles / duration_s)
    f_max = 1.0 / min_period_s
    df = 1.0 / (oversample * duration_s)
    freqs = np.arange(f_min, f_max, df)
    if len(freqs) < 3:
        return None, None, {"reason": "period search range too narrow for the "
                                       "rayleigh fallback (after the minimum-cycles "
                                       "cutoff)", "method": "rayleigh"}

    ang = 2 * np.pi * np.outer(t, freqs)
    c = np.cos(ang).sum(axis=0)
    s = np.sin(ang).sum(axis=0)
    r = np.sqrt(c ** 2 + s ** 2) / n
    z = 2 * n * r ** 2

    n_eff = max((f_max - f_min) * duration_s, 1.0)
    z_thresh = 2 * np.log(n_eff / alpha)

    is_peak = np.ones(len(freqs), dtype=bool)
    is_peak[1:] &= z[1:] >= z[:-1]
    is_peak[:-1] &= z[:-1] >= z[1:]
    candidates = np.where(is_peak & (z > z_thresh))[0]
    if len(candidates) == 0:
        return None, None, {"reason": "no significant periodicity found (rayleigh)",
                            "z_thresh": round(float(z_thresh), 1),
                            "n_flagged_events": n, "method": "rayleigh"}

    best = int(candidates[np.argmax(z[candidates])])
    period_s = 1.0 / freqs[best]
    phase_angle = np.arctan2(s[best], c[best])
    phase_s = ((phase_angle / (2 * np.pi)) % 1.0) * period_s
    return period_s, phase_s, {"z": round(float(z[best]), 1),
                               "z_thresh": round(float(z_thresh), 1),
                               "n_flagged_events": n, "method": "rayleigh"}


def _refine_period_phase(flag_ts: np.ndarray, markers: np.ndarray, period_s: float,
                         mid_frac: float, n_iter: int = 3):
    """Refine a coarse period so reconstructed markers do not drift.

    Each cycle's event-cluster centre (median event time within the cycle)
    is regressed on cycle number with outlier rejection; the slope is the
    period. Returns ``(markers, period_s)`` or None if too few cycles have
    events.
    """
    flag_ts = np.sort(np.asarray(flag_ts, dtype=float))
    for _ in range(n_iter):
        k, c = [], []
        for i, m in enumerate(markers):
            a, b = np.searchsorted(flag_ts, [m, m + period_s])
            if b - a >= 3:
                k.append(i)
                c.append(np.median(flag_ts[a:b]))
        if len(k) < 20:
            return None
        k, c = np.asarray(k, float), np.asarray(c)
        keep = np.ones(len(k), bool)
        for _ in range(3):
            slope, icpt = np.polyfit(k[keep], c[keep], 1)
            res = c - (icpt + slope * k)
            mad = np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-9
            keep = np.abs(res - np.median(res[keep])) < 5 * 1.4826 * mad
        if not (0.9 * period_s < slope < 1.1 * period_s):
            return None
        period_s = float(slope)
        markers = icpt + slope * np.arange(len(markers)) - mid_frac * period_s
    return markers, period_s


def detect_pseudo_stim_times(sig: np.ndarray, ts: np.ndarray, fs: float,
                             curv_thresh: float | None = None,
                             min_events: int = DEFAULT_MIN_EVENTS_FOR_PERIODICITY,
                             min_period_s: float = DEFAULT_MIN_PERIOD_S,
                             max_period_s: float = DEFAULT_MAX_PERIOD_S,
                             bin_ms: float = DEFAULT_AUTOCORR_BIN_MS
                             ) -> tuple[np.ndarray | None, dict]:
    """Reconstruct stim-onset times from the artifact itself (no markers).

    The period is estimated from the autocorrelation of curvature events
    (Rayleigh folding as a fallback for sparse events), the markers are
    phased to put the artifact in the middle of the calibration search range,
    and the period is then refined to remove drift.

    Returns ``(pseudo_stim_times, diagnostics)``, or ``(None, diagnostics)``
    when no periodic artifact is found.
    """
    sig = np.asarray(sig, dtype=float)
    ts = np.asarray(ts, dtype=float)
    if curv_thresh is None:
        curv_thresh, _ = resolve_curv_thresh(sig, fs)
    _, flagged = _curvature_events(sig, curv_thresh, pad=0)
    if len(flagged) < min_events:
        return None, {"reason": f"too few curvature-flagged events "
                                f"({len(flagged)} < {min_events}) to test for periodicity",
                      "n_flagged_events": len(flagged)}

    flag_ts = ts[flagged]
    duration_s = float(ts[-1] - ts[0])
    period_s, diag = _estimate_period(flag_ts - ts[0], duration_s, bin_ms=bin_ms,
                                      min_period_s=min_period_s, max_period_s=max_period_s)
    diag["n_flagged_events"] = len(flagged)
    center_phase_s = None
    if period_s is not None:
        center_phase_s = _dominant_phase(flag_ts, ts[0], period_s, bin_ms=bin_ms)

    if period_s is None or center_phase_s is None:
        period_s, center_phase_s, rdiag = _rayleigh_period_phase(
            flag_ts, duration_s, min_period_s=min_period_s, max_period_s=max_period_s)
        if period_s is None:
            diag["primary_reason"] = diag.get("reason")
            diag["rayleigh"] = rdiag
            diag["reason"] = rdiag.get("reason", "no periodicity found")
            return None, diag
        diag = {**diag, "primary_reason": diag.get("reason"), "rayleigh": rdiag}
        diag.pop("reason", None)

    arc_center_s = _artifact_arc_center(flag_ts, ts[0], period_s)
    if arc_center_s is not None:
        diag["phase_shift_ms"] = round(float(
            ((arc_center_s - center_phase_s + period_s / 2) % period_s - period_s / 2) * 1000), 1)
        center_phase_s = arc_center_s

    # Centre the artifact in calibrate_artifact_window's search range.
    mid_frac = 0.95 / 2.0 - DEFAULT_CALIB_PRE_FRAC
    t0 = ts[0] + center_phase_s - mid_frac * period_s
    pseudo_stim_times = np.arange(t0, ts[-1], period_s)
    pseudo_stim_times = pseudo_stim_times[pseudo_stim_times >= ts[0]]
    refined = _refine_period_phase(flag_ts, pseudo_stim_times, period_s, mid_frac)
    if refined is not None:
        diag["period_coarse_s"] = period_s
        pseudo_stim_times, period_s = refined
    diag["period_s"] = period_s
    diag["n_pseudo_markers"] = len(pseudo_stim_times)
    return pseudo_stim_times, diag


def remove_stim_artifact_no_markers(sig: np.ndarray, ts: np.ndarray, fs: float,
                                    curv_thresh: float | None = None,
                                    pad: int | None = None,
                                    detection_kwargs: dict | None = None,
                                    calibration_kwargs: dict | None = None,
                                    phase_locked: bool = False,
                                    phase_locked_kwargs: dict | None = None,
                                    template: bool = True
                                    ) -> tuple[np.ndarray, dict]:
    """Remove stimulation artifact without stim markers.

    Stim-onset times are reconstructed with ``detect_pseudo_stim_times`` and
    passed to ``remove_stim_artifact``. Same return value, plus a
    ``detection`` entry in the diagnostics. If no periodic artifact is found
    the signal is returned unchanged.
    """
    thresh_info = {"curv_thresh": curv_thresh, "rule": "explicit"}
    if curv_thresh is None:
        curv_thresh, thresh_info = resolve_curv_thresh(sig, fs)
    pseudo_stim_times, detection = detect_pseudo_stim_times(
        sig, ts, fs, curv_thresh=curv_thresh, **(detection_kwargs or {}))
    if pseudo_stim_times is None:
        return np.asarray(sig, dtype=float).copy(), {
            "n_cycles": 0, "n_spikes": 0, "window_ms": None,
            "calibration": None, "detection": detection, "threshold": thresh_info}

    cleaned, diag = remove_stim_artifact(
        sig, ts, pseudo_stim_times, fs, curv_thresh=curv_thresh, pad=pad,
        calibration_kwargs=calibration_kwargs,
        phase_locked=phase_locked, phase_locked_kwargs=phase_locked_kwargs,
        template=template)
    diag["detection"] = detection
    diag["threshold"] = thresh_info
    return cleaned, diag
