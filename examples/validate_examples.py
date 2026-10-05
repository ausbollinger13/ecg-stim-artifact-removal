"""Clean every example recording (with and without stim markers), print the
validation metrics described in docs/validation.md and save a before/after
figure to docs/figures/.

    python examples/validate_examples.py
"""
import glob
import os
import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from stim_artifact_removal import clean_ecg, ecg_hrv, load_ecg_file

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG_DIR = os.path.join(os.path.dirname(HERE), "docs", "figures")
CONDS = ["sham", "cymba", "cervical", "back"]
COLOR = {"sham": "#888888", "cymba": "#13B4EC", "cervical": "#E8A33D", "back": "#9200FF"}


def stim_locked_ratio(x, ts, stim, fs, band, rng, n_null=60):
    """Peak-to-peak of the stim-locked average of `x` in `band`, divided by
    the 99th percentile of the same quantity at random times. ~1 means
    nothing stimulation-locked is left."""
    if len(stim) < 20:
        return np.nan
    kind = "bandpass" if isinstance(band, tuple) else "high"
    h = sosfiltfilt(butter(4, band, kind, fs=fs, output="sos"), x)
    period = float(np.median(np.diff(stim)))
    a, b = int(-0.05 * period * fs), int(0.95 * period * fs)
    idx = np.searchsorted(ts, stim)
    idx = idx[(idx + a >= 0) & (idx + b < len(h))]
    avg = lambda ii: np.mean([h[i + a:i + b] for i in ii], axis=0)
    null = [np.ptp(avg(rng.integers(-a, len(h) - b, len(idx)))) for _ in range(n_null)]
    return float(np.ptp(avg(idx)) / np.percentile(null, 99))


def main():
    rng = np.random.default_rng(0)
    rows, traces = [], {}
    for path in sorted(glob.glob(os.path.join(DATA, "*.mat"))):
        name = os.path.splitext(os.path.basename(path))[0]
        subject, cond = name.split("_")
        rec = load_ecg_file(path, interactive=False)
        stim_path = os.path.join(DATA, name + "_stim_times.csv")
        stim = np.loadtxt(stim_path, ndmin=1) if os.path.exists(stim_path) else np.array([])
        for markers in (False, True):
            out = clean_ecg(rec["sig"], rec["ts"], rec["fs"],
                            stim_times=stim if markers and len(stim) else None)
            pre, cl, ts, fs = out["ecg_preprocessed"], out["ecg"], out["ts"], out["fs"]
            h0, h1 = ecg_hrv(pre, fs), ecg_hrv(cl, fs)
            rows.append({
                "subject": subject, "cond": cond, "markers": markers,
                "rec_fs": rec["fs"], "mode": out["mode"],
                "window_ms": None if out["diagnostics"]["window_ms"] is None
                else tuple(int(round(w)) for w in out["diagnostics"]["window_ms"]),
                "spike_raw": stim_locked_ratio(pre, ts, stim, fs, 100.0, rng),
                "spike_clean": stim_locked_ratio(cl, ts, stim, fs, 100.0, rng),
                "mid_raw": stim_locked_ratio(pre, ts, stim, fs, (10.0, 100.0), rng),
                "mid_clean": stim_locked_ratio(cl, ts, stim, fs, (10.0, 100.0), rng),
                "beats_raw": h0.get("n_peaks"), "beats_clean": h1.get("n_peaks"),
                "RMSSD_raw": h0.get("RMSSD"), "RMSSD_clean": h1.get("RMSSD"),
            })
            if not markers:
                traces[subject, cond] = (ts, pre, cl, stim)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(2).to_string(index=False))
    figure(traces, os.path.join(FIG_DIR, "examples_grid.png"))
    return df


def figure(traces, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    subjects = sorted({s for s, _ in traces})
    fig, axes = plt.subplots(len(subjects), len(CONDS), figsize=(17, 3.0 * len(subjects)),
                             squeeze=False)
    for i, s in enumerate(subjects):
        for j, c in enumerate(CONDS):
            ax = axes[i, j]
            ts, pre, cl, stim = traces[s, c]
            t0 = stim[20] if len(stim) > 20 else ts[0] + 20.0
            m = (ts >= t0 - 0.1) & (ts < t0 + 1.9)
            ax.plot(ts[m] - t0, pre[m], color="#AAAAAA", lw=0.5, label="before")
            ax.plot(ts[m] - t0, cl[m], color=COLOR[c], lw=0.9, label="after")
            lo, hi = np.percentile(cl, [0.05, 99.95])
            ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))
            if i == 0:
                ax.set_title(c.capitalize())
            if j == 0:
                ax.set_ylabel(f"{s}\nECG (amplifier units)")
            if i == len(subjects) - 1:
                ax.set_xlabel("time from stim onset (s)")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("clean_ecg without stim markers: before (grey) and after (colour); "
                 "y-axis scaled to the cleaned trace")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
