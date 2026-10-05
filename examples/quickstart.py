"""Minimal example: clean one example recording and compare HRV.

    python examples/quickstart.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from stim_artifact_removal import clean_ecg, ecg_hrv, load_ecg_file

HERE = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(HERE, "data", "subjectA_cymba.mat")

rec = load_ecg_file(path, interactive=False)          # finds `ecg` and `fs` in the file
print(f"{rec['source']}: {len(rec['sig'])} samples at {rec['fs']:g} Hz")

# No stim markers: the artifact's period and phase are found from the signal.
out = clean_ecg(rec["sig"], rec["ts"], rec["fs"])
d = out["diagnostics"]
print(f"mode: {out['mode']}")
print(f"artifact window: {d['window_ms']} ms per cycle, {d['n_cycles']} cycles, "
      f"{d['n_spikes']} spike samples removed")

# With stim markers (optional), if you have them.
stim = np.loadtxt(os.path.join(HERE, "data", "subjectA_cymba_stim_times.csv"))
out_mk = clean_ecg(rec["sig"], rec["ts"], rec["fs"], stim_times=stim)
print(f"with markers: window {out_mk['diagnostics']['window_ms']} ms")

before = ecg_hrv(out["ecg_preprocessed"], out["fs"])
after = ecg_hrv(out["ecg"], out["fs"])
for k in ("n_peaks", "MeanNN", "SDNN", "RMSSD"):
    print(f"  {k:8s} before {before.get(k, float('nan')):8.1f}   after {after.get(k, float('nan')):8.1f}")

t, fs = out["ts"], out["fs"]
m = (t >= 10) & (t < 13)
plt.figure(figsize=(10, 3.5))
plt.plot(t[m], out["ecg_preprocessed"][m], color="0.6", lw=0.6, label="before")
plt.plot(t[m], out["ecg"][m], color="C0", lw=0.9, label="after")
lo, hi = np.percentile(out["ecg"], [0.05, 99.95])
plt.ylim(lo - 0.2 * (hi - lo), hi + 0.2 * (hi - lo))
plt.xlabel("time (s)")
plt.ylabel("ECG (amplifier units)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
