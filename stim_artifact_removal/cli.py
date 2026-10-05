"""Command-line interface: clean one or more ECG files (or folders of them)
and write, for each, the cleaned signal, diagnostics, an HRV before/after
summary and a before/after figure.

    stim-artifact-removal --input my_ecg.xlsx
    stim-artifact-removal --input recordings/ --recursive --non-interactive
    stim-artifact-removal --input a.mat b.json c.xdf --out-format mat

(``python -m stim_artifact_removal`` works the same way.)

Omit --input and you'll be prompted to pick file(s) -- a native file-picker
dialog if a display is available, otherwise a list of matching files in the
current directory.

Supported input formats, auto-detected from the extension (see loaders.py
for exactly how each is read):
  .csv .txt .tsv .dat   delimited text, any common delimiter, header or not,
                        '#' comment lines allowed (a '# fs = 1000' comment
                        is used as the sampling rate)
  .xlsx .xlsm .xls      Excel -- the sheet with the most numbers is used;
                        a ("fs", 1000) row on another sheet is picked up
  .json .jsonl .ndjson  records, column arrays, nested objects, 2-D arrays
                        with channel labels, pandas-style exports
  .mat                  MATLAB (classic and v7.3): variables, structs,
                        2-D channel arrays, an fs variable
  .xdf                  LabStreamingLayer -- ECG stream/channel found by
                        label, then by heart-rhythm regularity

Stim markers are optional and usually absent: without them the marker-free
path (`remove_stim_artifact_no_markers`) runs. `--stim-times` (a CSV of
stim-onset timestamps) switches to marker-based cleaning; an XDF file's own
single Markers stream is tried too, with an automatic fall-back to
marker-free detection if those markers don't line up with any artifact.

Run `python -m stim_artifact_removal --help` for the full option list.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from . import loaders
from .pipeline import clean_ecg
from .hrv import ecg_hrv
from .preprocess import DEFAULT_TARGET_FS, DEFAULT_LINE_FREQ, DEFAULT_HP_CUTOFF

FILETYPE_PATTERN = " ".join(f"*{e}" for e in loaders.SUPPORTED_EXTENSIONS)
OUTPUT_SUFFIXES = ("_cleaned.csv", "_cleaned.xlsx", "_cleaned.mat", "_cleaned.json",
                   "_diagnostics.json", "stim_artifact_removal_summary.csv")


def _pick_file_interactively() -> list[str]:
    """Native file-picker dialog if a display is available (several files
    may be selected), else a numbered list of matching files in the current
    directory."""
    try:
        import tkinter
        from tkinter import filedialog
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        paths = filedialog.askopenfilenames(
            title="Select ECG file(s) to clean",
            filetypes=[("Supported ECG files", FILETYPE_PATTERN),
                      ("Excel", "*.xlsx *.xlsm *.xls"), ("Text / CSV", "*.csv *.txt *.tsv *.dat"),
                      ("JSON", "*.json *.jsonl *.ndjson"), ("MATLAB", "*.mat"),
                      ("XDF", "*.xdf"), ("All files", "*.*")])
        root.destroy()
        if paths:
            return list(paths)
        print("No file selected.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"(GUI file picker unavailable ({e}) -- falling back to a text prompt)")

    cwd = os.getcwd()
    candidates = sorted(f for f in os.listdir(cwd)
                        if os.path.splitext(f)[1].lower() in loaders.SUPPORTED_EXTENSIONS)
    if candidates:
        print(f"\nECG files found in {cwd}:")
        for i, f in enumerate(candidates):
            print(f"  [{i}] {f}")
        raw = input("Enter a number, or type a full file path: ").strip()
        if raw.isdigit() and int(raw) < len(candidates):
            return [os.path.join(cwd, candidates[int(raw)])]
        return [raw]
    return [input(f"Enter the path to an ECG file ({' '.join(loaders.SUPPORTED_EXTENSIONS)}): ").strip()]


def expand_inputs(inputs: list[str], recursive: bool = False) -> list[str]:
    """Files as given; folders expanded to the supported files inside them,
    skipping this tool's own outputs and *_stim_artifact_removed folders."""
    out = []
    for item in inputs:
        if os.path.isdir(item):
            walker = os.walk(item) if recursive else [(item, [], os.listdir(item))]
            for root, dirs, files in walker:
                dirs[:] = [d for d in dirs if not d.endswith("_stim_artifact_removed")]
                out += [os.path.join(root, f) for f in sorted(files)
                        if os.path.isfile(os.path.join(root, f))
                        and os.path.splitext(f)[1].lower() in loaders.SUPPORTED_EXTENSIONS
                        and not f.endswith(OUTPUT_SUFFIXES)]
        elif os.path.isfile(item):
            out.append(item)
        else:
            raise FileNotFoundError(item)
    return out


def _load_stim_times(path: str) -> np.ndarray:
    # Deliberately not `sep=None` auto-sniffing -- on a single unlabeled
    # numeric column it can misfire (varying digit groupings look like
    # different delimiters to the sniffer on different lines).
    df = pd.read_csv(path, header=None)
    col = 1 if df.shape[1] > 1 and df.iloc[:, 0].dtype == object else 0
    return df.iloc[:, col].to_numpy(dtype=np.float64)


def _make_figure(out_path: str, raw: np.ndarray, cleaned: np.ndarray, ts: np.ndarray,
                 fs: float, diag: dict, stem: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window_ms = diag.get("window_ms")
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))

    # Top: zoom in on an ACTUAL despiked cycle by finding where raw and
    # cleaned differ most -- not by assuming window_ms is an offset from
    # the start of the recording (it's an offset from each individual
    # cycle's own marker/reconstructed-marker time, which isn't otherwise
    # available here in marker-free mode).
    diff = np.abs(cleaned - raw)
    if window_ms is not None and diag.get("n_cycles", 0) > 0 and diff.max() > 0:
        center = int(np.argmax(diff))
        span = max(int(fs), int((window_ms[1] - window_ms[0]) * fs / 1000) + int(0.4 * fs))
        idx0 = max(0, center - span // 2)
        idx1 = min(len(raw), idx0 + span)
        axes[0].plot(ts[idx0:idx1], raw[idx0:idx1], color="#888888", lw=1.0, label="before")
        axes[0].plot(ts[idx0:idx1], cleaned[idx0:idx1], color="#13B4EC", lw=1.0, label="cleaned")
        axes[0].set_title(f"One despiked cycle (window={tuple(round(w) for w in window_ms)} ms "
                          f"from each cycle start)")
        axes[0].legend(fontsize=8)
    else:
        n = min(len(raw), int(5 * fs))
        axes[0].plot(ts[:n], raw[:n], color="#888888", lw=0.8, label="before")
        axes[0].plot(ts[:n], cleaned[:n], color="#13B4EC", lw=0.8, label="cleaned")
        axes[0].set_title("No artifact-like periodicity found -- first 5 s shown as-is")
        axes[0].legend(fontsize=8)

    # Bottom: the whole trace, downsampled for plotting only.
    step = max(1, len(raw) // 20000)
    axes[1].plot(ts[::step], raw[::step], color="#888888", lw=0.5, label="before")
    axes[1].plot(ts[::step], cleaned[::step], color="#13B4EC", lw=0.5, label="cleaned")
    axes[1].set_title("Full trace")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"stim_artifact_removal: {stem}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _write_cleaned(path_stem: str, fmt: str, ts, sig, cleaned, fs: float) -> str:
    # Both signal columns are AFTER preprocessing (decimated, filtered and
    # polarity-corrected): `ecg_preprocessed` is the input to cleaning, not
    # the file's untouched samples.
    cols = {"time": ts, "ecg_preprocessed": sig, "ecg_cleaned": cleaned}
    if fmt == "csv":
        out = path_stem + ".csv"
        pd.DataFrame(cols).to_csv(out, index=False)
    elif fmt == "xlsx":
        if len(ts) > 1_048_575:
            raise ValueError(f"{len(ts)} samples exceed Excel's 1,048,576-row limit -- "
                             f"use --out-format csv, mat, or json")
        out = path_stem + ".xlsx"
        with pd.ExcelWriter(out) as xw:
            pd.DataFrame(cols).to_excel(xw, sheet_name="data", index=False)
            pd.DataFrame({"key": ["fs"], "value": [fs]}).to_excel(xw, sheet_name="info", index=False)
    elif fmt == "mat":
        from scipy.io import savemat
        out = path_stem + ".mat"
        savemat(out, {**cols, "fs": fs}, do_compression=True)
    elif fmt == "json":
        out = path_stem + ".json"
        with open(out, "w") as fh:
            json.dump({"fs": fs, **{k: np.asarray(v).tolist() for k, v in cols.items()}}, fh)
    else:
        raise ValueError(f"unknown output format {fmt!r}")
    return out


def run(input_path: str, fs: float | None, stim_times_path: str | None,
       out_dir: str | None, target_fs: float, line_freq: float, hp_cutoff: float,
       curv_thresh: float | None, no_preprocess: bool, no_figures: bool, no_hrv: bool,
       interactive: bool = True, phase_locked: bool = False, channel=None, sheet=None,
       out_format: str = "csv", clean_rate: str = "native") -> dict:
    stem = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(input_path)),
                                       f"{stem}_stim_artifact_removed")
    os.makedirs(out_dir, exist_ok=True)

    loaded = loaders.load_ecg_file(input_path, fs=fs, interactive=interactive,
                                   channel=channel, sheet=sheet)
    raw_sig, raw_ts, raw_fs = loaded["sig"], loaded["ts"], loaded["fs"]
    embedded_stim_times = loaded.get("stim_times")
    print(f"Loaded {len(raw_sig)} samples ({len(raw_sig) / raw_fs:.1f} s) from "
         f"{input_path} (fs={raw_fs:.2f} Hz)")
    if embedded_stim_times is not None:
        print(f"File also provided {len(embedded_stim_times)} stim-marker timestamps")

    stim_times, mode_note = None, ""
    if stim_times_path:
        stim_times = _load_stim_times(stim_times_path)
        print(f"Using {len(stim_times)} stim markers from {stim_times_path} "
              f"(overrides any markers embedded in {input_path})")
    elif embedded_stim_times is not None and len(embedded_stim_times) >= 3:
        stim_times = embedded_stim_times
        print(f"Trying the {len(embedded_stim_times)} stim markers embedded in {input_path} "
              f"(marker-free detection is used instead if they don't line up with artifact)")
    else:
        print("No stim markers available -- running marker-free detection")
    if phase_locked:
        print("phase_locked=True: also despiking offsets reliably artifact-locked across cycles")

    out = clean_ecg(raw_sig, raw_ts, raw_fs, stim_times=stim_times, target_fs=target_fs,
                    line_freq=line_freq, hp_cutoff=hp_cutoff,
                    native_rate=(clean_rate == "native"), preprocess=not no_preprocess,
                    curv_thresh=curv_thresh, phase_locked=phase_locked)
    sig, cleaned, ts, proc_fs = out["ecg_preprocessed"], out["ecg"], out["ts"], out["fs"]
    diag, mode, inverted = out["diagnostics"], out["mode"], out["polarity_flipped"]
    th = diag.get("threshold") or {}
    print(f"Cleaned at {out['clean_fs']:.0f} Hz"
          + ("" if no_preprocess else f" ({line_freq:g} Hz notch, {hp_cutoff} Hz high-pass, "
                                      f"polarity {'flipped' if inverted else 'unchanged'})")
          + (f", output at {proc_fs:.0f} Hz" if proc_fs != out["clean_fs"] else ""))
    if th.get("curv_thresh") is not None:
        print(f"Curvature threshold: {th['curv_thresh']:.4g} -- {th.get('rule', '')}")
    print(f"Mode: {mode}")

    if diag["window_ms"] is None:
        print("No periodic stimulation artifact detected -- signal returned unchanged. "
              "See the diagnostics JSON for why (this is the correct outcome for "
              "genuinely clean data).")
    else:
        period = (diag.get("detection") or {}).get("period_s")
        tpl = diag.get("template") or {}
        print(f"Artifact found{f' (period {period:.4f} s)' if period else ''}: window "
              f"{tuple(round(w) for w in diag['window_ms'])} ms from each cycle start, "
              f"{diag['n_cycles']} cycles, {diag['n_spikes']} spike samples removed"
              + (f"; stim-locked template subtracted over {tpl['frac_offsets'] * 100:.0f}% "
                 f"of the cycle" if "frac_offsets" in tpl else ""))

    result = {"input": os.path.abspath(input_path), "source": loaded.get("source"),
              "fs": proc_fs, "clean_fs": out["clean_fs"], "mode": mode,
              "polarity_flipped": bool(inverted), "phase_locked": phase_locked,
              "diagnostics": diag}

    if not no_hrv:
        hrv_raw = ecg_hrv(sig, proc_fs)
        hrv_clean = ecg_hrv(cleaned, proc_fs)
        result["hrv_raw"] = hrv_raw
        result["hrv_cleaned"] = hrv_clean
        print(f"HRV before: SDNN={hrv_raw.get('SDNN', float('nan')):.1f}  "
             f"RMSSD={hrv_raw.get('RMSSD', float('nan')):.1f}  "
             f"n_peaks={hrv_raw.get('n_peaks', 'NA')}")
        print(f"HRV after:  SDNN={hrv_clean.get('SDNN', float('nan')):.1f}  "
             f"RMSSD={hrv_clean.get('RMSSD', float('nan')):.1f}  "
             f"n_peaks={hrv_clean.get('n_peaks', 'NA')}")

    out_data = _write_cleaned(os.path.join(out_dir, f"{stem}_cleaned"), out_format,
                              ts, sig, cleaned, proc_fs)
    result["output"] = out_data
    print(f"Wrote {out_data}")

    def _json_safe(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(str(type(o)))

    out_json = os.path.join(out_dir, f"{stem}_diagnostics.json")
    with open(out_json, "w") as fh:
        json.dump(result, fh, indent=2, default=_json_safe)
    print(f"Wrote {out_json}")

    if not no_figures:
        out_png = os.path.join(out_dir, f"{stem}_before_after.png")
        _make_figure(out_png, sig, cleaned, ts, proc_fs, diag, stem)
        print(f"Wrote {out_png}")

    return result


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="stim-artifact-removal",
        description="Remove periodic electrical-stimulation artifact from ECG recordings "
                    "(no stim markers needed) and report HRV before/after.")
    p.add_argument("--input", nargs="+", default=None,
                  help="One or more ECG files and/or folders of them (" +
                       " ".join(loaders.SUPPORTED_EXTENSIONS) + "). Omit to be prompted.")
    p.add_argument("--recursive", action="store_true",
                  help="With a folder --input, also search its subfolders.")
    p.add_argument("--channel", default=None,
                  help="Column / variable / channel name (or 0-based index) holding the ECG -- "
                       "skips auto-detection.")
    p.add_argument("--sheet", default=None,
                  help="Excel sheet name (or 0-based index) -- skips auto-detection.")
    p.add_argument("--fs", type=float, default=None,
                  help="Sampling rate of --input, Hz. Needed only if the file has neither a time "
                       "column nor a sampling-rate value; overrides both if given.")
    p.add_argument("--stim-times", default=None,
                  help="Optional CSV of stim-onset timestamps (seconds, same time base as --input). "
                       "Omit to use marker-free detection.")
    p.add_argument("--out-dir", default=None,
                  help="Output directory (default: <input dir>/<stem>_stim_artifact_removed/). "
                       "With several inputs, each gets its own <stem>_stim_artifact_removed/ "
                       "subfolder inside it.")
    p.add_argument("--out-format", choices=["csv", "xlsx", "mat", "json"], default="csv",
                  help="Format of the cleaned-signal output file (default csv).")
    p.add_argument("--target-fs", type=float, default=DEFAULT_TARGET_FS,
                  help=f"Output sampling rate, Hz (default {DEFAULT_TARGET_FS}); cleaning itself "
                       f"runs at the file's own rate unless --clean-rate target")
    p.add_argument("--clean-rate", choices=["native", "target"], default="native",
                  help="native (default): clean at the recording's own sampling rate, then "
                       "decimate to --target-fs -- spikes and QRS separate far better at high "
                       "rates. target: decimate first, then clean.")
    p.add_argument("--line-freq", type=float, default=DEFAULT_LINE_FREQ,
                  help=f"Mains hum frequency for the notch filter, Hz -- 60 (NA) or 50 "
                       f"(most elsewhere) (default {DEFAULT_LINE_FREQ})")
    p.add_argument("--hp-cutoff", type=float, default=DEFAULT_HP_CUTOFF,
                  help=f"High-pass cutoff for baseline wander, Hz (default {DEFAULT_HP_CUTOFF})")
    p.add_argument("--curv-thresh", type=float, default=None,
                  help="Curvature threshold for spike detection. Default: set from the "
                       "recording's own QRS curvature (resolve_curv_thresh). Lower it by hand "
                       "only if artifact is still visible after cleaning; see docs/method.md.")
    p.add_argument("--phase-locked", action="store_true",
                  help="Also despike offsets found reliably artifact-locked across many cycles, "
                       "even where a single cycle's own curvature is too weak to trust alone -- "
                       "recommended for weak-coupling sites where cleaning barely moves the "
                       "signal. Off by default; see docs/method.md.")
    p.add_argument("--no-preprocess", action="store_true",
                  help="Skip decimation/notch/high-pass/polarity correction -- use if --input is "
                       "already a conditioned signal.")
    p.add_argument("--no-figures", action="store_true", help="Skip saving the before/after PNG.")
    p.add_argument("--no-hrv", action="store_true", help="Skip NeuroKit2 HRV computation.")
    p.add_argument("--non-interactive", action="store_true",
                  help="Never prompt: when a stream/sheet/channel is ambiguous the best guess is "
                       "used (and printed). Recommended for folders/batches.")
    args = p.parse_args(argv)

    inputs = args.input
    if inputs is None:
        if args.non_interactive:
            print("ERROR: --input is required with --non-interactive", file=sys.stderr)
            sys.exit(1)
        inputs = _pick_file_interactively()
    try:
        paths = expand_inputs(inputs, args.recursive)
    except FileNotFoundError as e:
        print(f"ERROR: not found: {e}", file=sys.stderr)
        sys.exit(1)
    if not paths:
        print("ERROR: no supported files found", file=sys.stderr)
        sys.exit(1)

    batch = len(paths) > 1
    summary, n_failed = [], 0
    for i, path in enumerate(paths):
        if batch:
            print(f"\n{'=' * 70}\n[{i + 1}/{len(paths)}] {path}\n{'=' * 70}")
        stem = os.path.splitext(os.path.basename(path))[0]
        out_dir = (os.path.join(args.out_dir, f"{stem}_stim_artifact_removed")
                   if (batch and args.out_dir) else args.out_dir)
        row = {"file": path}
        try:
            res = run(path, args.fs, args.stim_times, out_dir, args.target_fs,
                     args.line_freq, args.hp_cutoff, args.curv_thresh, args.no_preprocess,
                     args.no_figures, args.no_hrv, interactive=not args.non_interactive,
                     phase_locked=args.phase_locked, channel=args.channel, sheet=args.sheet,
                     out_format=args.out_format, clean_rate=args.clean_rate)
            d = res["diagnostics"]
            row.update(status="ok", source=res.get("source"), mode=res["mode"],
                       artifact_found=d["window_ms"] is not None, window_ms=d["window_ms"],
                       n_spikes=d["n_spikes"], output=res.get("output"))
            for tag in ("raw", "cleaned"):
                h = res.get(f"hrv_{tag}") or {}
                row.update({f"{tag}_n_peaks": h.get("n_peaks"), f"{tag}_SDNN": h.get("SDNN"),
                            f"{tag}_RMSSD": h.get("RMSSD")})
        except Exception as e:
            n_failed += 1
            row.update(status=f"ERROR: {e}")
            print(f"ERROR: {e}", file=sys.stderr)
        summary.append(row)

    if batch:
        base = args.out_dir or os.path.dirname(os.path.abspath(paths[0]))
        os.makedirs(base, exist_ok=True)
        out = os.path.join(base, "stim_artifact_removal_summary.csv")
        pd.DataFrame(summary).to_csv(out, index=False)
        print(f"\n{len(paths) - n_failed}/{len(paths)} file(s) processed; summary: {out}")
    if n_failed == len(paths):
        sys.exit(1)


if __name__ == "__main__":
    main()
