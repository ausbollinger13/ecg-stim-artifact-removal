# Usage

## Command line

```bash
stim-artifact-removal --input my_ecg.xlsx
stim-artifact-removal --input a.mat b.json c.xdf --out-format mat
stim-artifact-removal --input recordings/ --recursive --non-interactive
stim-artifact-removal                      # pick file(s) in a dialog
```

`python -m stim_artifact_removal` is equivalent.

| option | meaning |
|---|---|
| `--input` | files and/or folders (omit to be prompted) |
| `--recursive` | also search subfolders of a folder input |
| `--channel NAME_OR_INDEX` | ECG column/variable/channel; skips auto-detection |
| `--sheet NAME_OR_INDEX` | Excel sheet; skips auto-detection |
| `--fs HZ` | sampling rate, if the file has neither a time column nor an fs value (overrides both) |
| `--stim-times FILE.csv` | stim-onset times (s, same clock as the recording), one per line, no header |
| `--out-dir DIR` | output folder (default `<input dir>/<stem>_stim_artifact_removed/`) |
| `--out-format csv\|xlsx\|mat\|json` | cleaned-signal file format (default csv) |
| `--target-fs HZ` | output sampling rate (default 1000) |
| `--clean-rate native\|target` | clean at the recording's own rate (default) or after decimating |
| `--line-freq HZ` | mains frequency for the notch: 60 (default) or 50 |
| `--hp-cutoff HZ` | baseline high-pass cutoff (default 0.5) |
| `--curv-thresh X` | fixed curvature threshold instead of the automatic one |
| `--phase-locked` | also remove weak but consistently stim-locked samples (see method.md) |
| `--no-preprocess` | input is already filtered and upright |
| `--no-figures`, `--no-hrv` | skip the figure / HRV |
| `--non-interactive` | never prompt; best guesses are used and printed |

### Outputs

For each input, in its output folder:

- `<stem>_cleaned.<fmt>`: `time`, `ecg_preprocessed` (the signal before
  cleaning, after polarity correction, filtering and decimation) and
  `ecg_cleaned`.
- `<stem>_diagnostics.json`: which stream/sheet/channel was used, whether
  markers or marker-free detection ran, polarity, the artifact window,
  thresholds, and HRV before/after.
- `<stem>_before_after.png`: one cleaned cycle zoomed in, plus the full
  trace.
- With several inputs: `stim_artifact_removal_summary.csv`, one row per
  file.

### How inputs are read

Every format is reduced to named 1-D arrays (table columns, MATLAB
variables or struct fields, JSON keys; 2-D arrays are split into
channels), plus any scalar values. Then:

- **Time.** A column named like `time`, `timestamp`, `t` or `seconds`. A
  name containing `ms` is divided by 1000, and spacing implying an ECG rate
  below 20 Hz is assumed to be milliseconds. Datetime columns are converted
  to seconds. A sample-index column is not treated as time.
- **Sampling rate.** `--fs` if given; else a scalar named `fs`, `srate`,
  `sampling_rate`, `Fs`, …; else inferred from the time column.
- **ECG channel.** `--channel` if given; else a name containing
  `ecg`/`ekg` or a lead name (`II`, `V1`, …); else the only numeric
  column; else the channel with the most regular, plausible heart rhythm.

Format notes:

| format | notes |
|---|---|
| `.csv .txt .tsv .dat` | any common delimiter, header optional, `#` comments allowed (`# fs = 1000` is read as the sampling rate) |
| `.xlsx .xlsm .xls` | the sheet with the most numbers is the data; an `("fs", 1000)` row on another sheet is picked up. Needs `openpyxl` (`xlrd` for `.xls`) |
| `.json .jsonl .ndjson` | records, `{column: [values]}`, nested objects, a 2-D array with a channel-label list, pandas exports, JSON Lines |
| `.mat` | classic and v7.3 (needs `h5py`); variables, structs, 2-D channel arrays. MATLAB `table` objects cannot be read outside MATLAB: save the columns as variables |
| `.xdf` | needs `pyxdf`; the ECG stream/channel is found by label, else by heart-rhythm regularity. A single Markers stream is tried as stim markers |

If the markers (from `--stim-times` or an XDF Markers stream) don't line
up with any artifact, for example a stream that mixes stim onsets with
task events, cleaning falls back to marker-free detection. The fallback is
reported in `mode`.

## Python

```python
from stim_artifact_removal import clean_ecg

out = clean_ecg(raw, ts, fs, stim_times=None, target_fs=1000, line_freq=60)
```

| key | content |
|---|---|
| `ecg` | cleaned ECG at `fs` |
| `ecg_preprocessed` | the same signal without artifact removal, for comparison |
| `ts`, `fs` | output timestamps and sampling rate |
| `clean_fs` | rate at which cleaning ran (the input rate by default) |
| `mode` | `"markers"`, `"marker-free"` or `"marker-free (markers did not line up with artifact)"` |
| `polarity_flipped` | whether the signal was inverted to make R waves positive |
| `diagnostics` | artifact window (`window_ms`, None if no artifact was found), `n_cycles`, `n_spikes`, `threshold`, `template`, `calibration`, `detection` |

Other public functions:

- `load_ecg_file(path, fs=None, interactive=True, channel=None, sheet=None)`
  returns `{sig, ts, fs, stim_times, source}`.
- `ecg_hrv(sig, fs)` returns NeuroKit2 time-domain HRV
  (`n_peaks, MeanNN, SDNN, RMSSD, pNN50`).
- `remove_stim_artifact(sig, ts, stim_times, fs)` and
  `remove_stim_artifact_no_markers(sig, ts, fs)` are the lower-level
  cleaners. They take an already-conditioned signal and clean it at the
  rate given. Unlike `clean_ecg`, `remove_stim_artifact` does not fall
  back to marker-free detection: check `diag["window_ms"] is None`.
- `resolve_curv_thresh(sig, fs)` returns the automatic threshold and how
  it was chosen.

## Recommendations

- **Record and clean at a high sampling rate.** A QRS's curvature falls
  with the square of the sampling rate while a spike's does not, so spikes
  and QRS separate far better at 5 kHz than at 1 kHz. Very high rates can
  be reduced first to save time: for a 25 kHz recording, cleaning at
  5 kHz (`raw5, ts5, fs5 = decimate_to(raw, ts, fs, 5000)`, then
  `clean_ecg(raw5, ts5, fs5)`) was as good as native and about 6× faster.
  Cleaning at 1 kHz was not.
- **Record enough stimulation cycles.** Calibration and template
  subtraction need ≥ 20 cycles. Weak artifact is detected more reliably
  without markers when there are more cycles; the validation recordings
  had 60-150.
- **Check `mode` and `window_ms`.** If `window_ms` is None, no periodic
  artifact was found and the signal was returned unchanged. That is the
  expected result for Sham or clean data.
