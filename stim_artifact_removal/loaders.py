"""Generic, format-flexible ECG file loading for the CLI: CSV, TXT/TSV/DAT,
Excel (XLSX/XLS), JSON (and JSON Lines), XDF (LabStreamingLayer), and MAT
(MATLAB, classic and v7.3) files.

Each loader returns a dict:
    {"sig": ndarray, "ts": ndarray, "fs": float, "stim_times": ndarray|None,
     "source": str}

`source` says which stream/sheet/variable/column the ECG came from, so the
choice is never silent.

HOW EVERY NON-XDF FORMAT IS READ
    Each file is reduced to a set of NAMED 1-D ARRAYS -- table columns (CSV,
    TXT, Excel, JSON records), or variables / struct fields (MAT, JSON
    objects), with 2-D arrays split into one array per channel (named from a
    `labels`/`channels`-style variable when the file has one) -- plus any
    scalar values. One shared picker then chooses:
      * time: a column/variable named like time/timestamp/t/seconds; a
        datetime column is converted to seconds; a name mentioning ms is
        divided by 1000; a column whose spacing implies an ECG sampling rate
        below 20 Hz is assumed to be milliseconds (printed when it happens).
        A sample-index column is NOT treated as time.
      * sampling rate: `--fs` if given, else a scalar named fs / srate /
        sampling_rate / sample_rate / Fs, else inferred from the time
        column. `--fs` disagreeing with the file's own time column by >1%
        is printed.
      * ECG: `channel=` if given (name or index), else a name containing
        ecg/ekg (or a lead name like "II"), else the only numeric candidate,
        else the channel with the most regular plausible heart rhythm
        (`_auto_pick_ecg_channel`). Interactive mode asks when the choice
        is not a clear name match.

XDF markers are used only when exactly one Markers stream exists (or the
user picks one interactively). If those markers turn out not to be stim
onsets (e.g. mixed event codes), ``clean_ecg`` falls back to marker-free
detection automatically. Other formats never supply markers; pass
``--stim-times`` if you have them.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

SUPPORTED_EXTENSIONS = (".csv", ".txt", ".tsv", ".dat", ".xlsx", ".xlsm", ".xls",
                        ".json", ".jsonl", ".ndjson", ".xdf", ".mat")

TIME_NAMES = {"time", "t", "timestamp", "timestamps", "time_s", "times", "ts",
              "seconds", "sec", "secs", "time_sec", "elapsed", "elapsed_time",
              "time_ms", "ms", "msec", "milliseconds", "timestamp_ms"}
INDEX_NAMES = {"sample", "samples", "index", "idx", "n", "sample_index", "sample_number",
               "unnamed: 0"}
FS_NAMES = {"fs", "srate", "sampling_rate", "sample_rate", "samplerate", "sampling_frequency",
            "sample_frequency", "samplingrate", "sfreq", "nominal_srate", "sampling_rate_hz",
            "fs_hz", "sample_rate_hz"}
FS_NAMES_WHOLE_ONLY = {"freq", "hz", "sr", "rate"}   # too generic to match as a suffix
LABEL_NAMES = {"labels", "label", "channels", "channel_names", "chan_names", "channel_labels",
               "chanlabels", "ch_names", "names", "leads", "columns"}
ECG_NAME_TOKENS = ("ecg", "ekg", "electrocardio")
LEAD_NAMES = {"i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6",
              "lead_i", "lead_ii", "lead_iii", "lead i", "lead ii", "lead iii"}
MIN_ECG_FS = 20.0


def load_ecg_file(path: str, fs: float | None = None, interactive: bool = True,
                  channel=None, sheet=None) -> dict:
    """Load one ECG trace from any supported file. `channel` (column/variable
    /channel name, or integer index) and `sheet` (Excel sheet name or index)
    skip the corresponding auto-detection."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".txt", ".tsv", ".dat"):
        arrays, scalars, source = _read_delimited(path)
    elif ext in (".xlsx", ".xlsm", ".xls"):
        arrays, scalars, source = _read_excel(path, sheet, interactive)
    elif ext in (".json", ".jsonl", ".ndjson"):
        arrays, scalars, source = _read_json(path)
    elif ext == ".mat":
        arrays, scalars, source = _read_mat(path)
    elif ext == ".xdf":
        return _load_xdf(path, interactive, channel)
    else:
        raise ValueError(f"Unsupported file type {ext!r} -- supported: {SUPPORTED_EXTENSIONS}")
    return _from_named_arrays(arrays, scalars, fs, interactive, channel, source)


# ══════════════════════════════════════════════════════════════════════════
#  Shared picker: named 1-D arrays (+ scalars) -> sig, ts, fs
# ══════════════════════════════════════════════════════════════════════════

def _norm(name) -> str:
    """'Time (ms)' -> 'time_ms', 'ECG Lead-II' -> 'ecg_lead_ii'."""
    return re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")


def _leaf(name) -> str:
    """Normalized last path component: 'meta.Time' -> 'time', 'S.fs' -> 'fs'."""
    return _norm(re.split(r"\.(?=[A-Za-z_])", str(name))[-1])


def _is_numeric(a: np.ndarray) -> bool:
    return np.issubdtype(a.dtype, np.number) or np.issubdtype(a.dtype, np.bool_)


def _to_float(a) -> np.ndarray | None:
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.datetime64):
        return None
    if _is_numeric(a):
        return a.astype(np.float64)
    try:
        return np.asarray([float(v) for v in a], dtype=np.float64)
    except (TypeError, ValueError):
        return None


def _datetime_seconds(a) -> np.ndarray | None:
    """Seconds since the first sample, if `a` is (or parses as) datetimes."""
    import pandas as pd
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.datetime64):
        t = pd.to_datetime(a)
    elif a.dtype == object and len(a) and not isinstance(a[0], (int, float, np.number)):
        try:
            t = pd.to_datetime(pd.Series(a), errors="raise")
        except (ValueError, TypeError):
            return None
    else:
        return None
    t = pd.Series(t)
    return (t - t.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64)


def _is_fs_name(name) -> bool:
    leaf = _leaf(name)
    toks = leaf.split("_")
    return leaf in FS_NAMES_WHOLE_ONLY or any("_".join(toks[i:]) in FS_NAMES
                                             for i in range(len(toks)))


def _find_fs(scalars: dict) -> tuple[float | None, str | None]:
    for k, v in scalars.items():
        if _is_fs_name(k):
            try:
                val = float(np.asarray(v).squeeze())
            except (TypeError, ValueError):
                continue
            if val > 0:
                return val, k
    return None, None


def _time_from(name: str, raw) -> tuple[np.ndarray | None, str]:
    """(time in seconds, note) for a candidate time array, or (None, reason)."""
    secs = _datetime_seconds(raw)
    if secs is not None:
        return secs, "datetime converted to seconds"
    t = _to_float(raw)
    if t is None or len(t) < 3 or not np.all(np.isfinite(t)):
        return None, "not numeric"
    if not t[-1] > t[0] or np.mean(np.diff(t) < 0) > 0.01:
        return None, "not increasing"
    dt = (t[-1] - t[0]) / (len(t) - 1)
    n = _leaf(name)
    if "ms" in n.split("_") or "msec" in n or "millis" in n:
        return t / 1000.0, "milliseconds (from its name) converted to seconds"
    if 1.0 / dt < MIN_ECG_FS:
        return t / 1000.0, (f"spacing implies {1.0 / dt:.2f} Hz, implausible for ECG -- "
                            f"assumed milliseconds and converted to seconds")
    return t, "seconds"


def _from_named_arrays(arrays: dict, scalars: dict, fs: float | None, interactive: bool,
                       channel, source: str) -> dict:
    if not arrays:
        raise ValueError(f"no numeric data found in {source}")
    names = list(arrays)
    n_main = max(len(a) for a in arrays.values())

    # ---- time --------------------------------------------------------------
    time_key, ts, time_note = None, None, ""
    for k in names:
        if _leaf(k) in TIME_NAMES and len(arrays[k]) == n_main:
            ts, time_note = _time_from(k, arrays[k])
            if ts is not None:
                time_key = k
                break
    if time_key is None:   # a single datetime-typed column counts as time too
        for k in names:
            if len(arrays[k]) == n_main and _datetime_seconds(arrays[k]) is not None:
                time_key, ts, time_note = k, _datetime_seconds(arrays[k]), "datetime converted to seconds"
                break
    if time_key is None:
        # Headerless numeric files: a generically-named FIRST column that
        # rises at a near-constant step is the time axis.
        k0 = names[0]
        t0 = _to_float(arrays[k0]) if re.fullmatch(r"col\d+|unnamed_\d+|\d+", _leaf(k0)) else None
        if t0 is not None and len(t0) == n_main and len(t0) > 10:
            d = np.diff(t0)
            if np.all(d > 0) and np.std(d) < 0.01 * np.mean(d):
                ts, time_note = _time_from("", t0)
                if ts is not None:
                    time_key, time_note = k0, f"first column rises at a constant step -- {time_note}"
    if time_key is not None:
        # Timestamps coarser than the sampling (e.g. Excel datetimes rounded
        # to 1 ms at 5 kHz) repeat values; rebuild an even grid over the span.
        d = np.diff(ts)
        if np.mean(d == 0) > 0.05 or np.std(d) > 0.5 * np.mean(d):
            n_rep = int(np.sum(d == 0))
            ts = np.linspace(ts[0], ts[-1], len(ts))
            time_note += (f"; {n_rep} repeated timestamps (resolution coarser than the "
                          f"sampling) -- rebuilt as an even grid over the same span")
        print(f"Time: {time_key!r} ({time_note})")

    # ---- ECG candidates: numeric, full-length, not time/index -----------------
    candidates = {}
    for k in names:
        if k == time_key or _leaf(k) in INDEX_NAMES:
            continue
        a = _to_float(arrays[k])
        if a is None or len(a) != n_main or np.nanstd(a) == 0:
            continue
        candidates[k] = a
    if not candidates:
        raise ValueError(f"could not find a numeric signal column/variable in {source} "
                         f"(found: {names})")

    # ---- sampling rate -------------------------------------------------------
    file_fs, fs_key = _find_fs(scalars)
    inferred = (len(ts) - 1) / float(ts[-1] - ts[0]) if ts is not None else None
    if fs is not None:
        if inferred is not None and abs(inferred - fs) / fs > 0.01:
            print(f"NOTE: --fs {fs:g} Hz differs from the file's time column "
                  f"({inferred:.2f} Hz) -- using --fs, and rebuilding time from it")
            ts = None
    elif inferred is not None:
        fs = inferred
    elif file_fs is not None:
        fs = file_fs
        print(f"Sampling rate: {fs:g} Hz (from {fs_key!r} in the file)")
    else:
        raise ValueError(f"{source} has no recognizable time column or sampling-rate value "
                         f"-- pass --fs")
    if ts is None:
        ts = np.arange(n_main) / fs

    # ---- ECG channel ---------------------------------------------------------
    keys = list(candidates)
    ecg_key, how = None, ""
    if channel is not None:
        if isinstance(channel, int) or (isinstance(channel, str) and channel.isdigit()):
            ecg_key, how = keys[int(channel)], "given by index"
        else:
            match = [k for k in keys if _norm(k) == _norm(channel)]
            if not match:
                raise ValueError(f"channel {channel!r} not found; available: {keys}")
            ecg_key, how = match[0], "given by name"
    if ecg_key is None:
        named = [k for k in keys if any(tok in _norm(k) for tok in ECG_NAME_TOKENS)]
        if not named:
            named = [k for k in keys if _leaf(k) in LEAD_NAMES]
        if len(named) == 1:
            ecg_key, how = named[0], "name match"
        elif len(keys) == 1:
            ecg_key, how = keys[0], "only numeric signal"
        else:
            pool = named or keys
            mat = np.column_stack([candidates[k] for k in pool])
            ecg_key = pool[_auto_pick_ecg_channel(mat, transposed=False, n_ch=len(pool), fs=fs)]
            how = (f"most regular plausible heart rhythm among {len(pool)} "
                   f"{'ECG-named ' if named else ''}candidates")
            if interactive:
                print("Candidate signals: " + ", ".join(f"[{i}] {k!r}" for i, k in enumerate(pool)))
                ecg_key = pool[_prompt_index("Which is the ECG?", pool.index(ecg_key), len(pool))]
                how = "chosen interactively"
    print(f"ECG: {ecg_key!r} ({how})")

    sig = candidates[ecg_key]
    ok = np.isfinite(sig) & np.isfinite(ts)
    if not ok.all():
        print(f"NOTE: dropped {int((~ok).sum())} non-finite samples")
        sig, ts = sig[ok], ts[ok]
    return {"sig": sig, "ts": ts, "fs": float(fs), "stim_times": None,
            "source": f"{source} :: {ecg_key}"}


# ══════════════════════════════════════════════════════════════════════════
#  Format readers -> (arrays, scalars, source)
# ══════════════════════════════════════════════════════════════════════════

def _table_to_arrays(df) -> dict:
    return {str(c): df[c].to_numpy() for c in df.columns}


def _header_row_is_data(columns) -> bool:
    try:
        [float(c) for c in columns]
        return True
    except (TypeError, ValueError):
        return False


def _read_delimited(path: str):
    """CSV/TXT/TSV/DAT, any common delimiter, with or without a header row.
    Leading '#'/'%' comment lines are skipped, but a `# fs = 1000`-style
    comment is still picked up as the sampling rate."""
    import pandas as pd
    scalars, n_comment, body = {}, 0, []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if not body and (not s or s.startswith(("#", "%"))):
                n_comment += 1
                m = re.match(r"[#%]+\s*([A-Za-z_ ]+?)\s*[:=]\s*([-+0-9.eE]+)", s)
                if m:
                    scalars[m.group(1).strip()] = float(m.group(2))
                continue
            if s:
                body.append(line)
            if len(body) >= 5:
                break
    if not body:
        raise ValueError(f"{path} has no data rows")
    sep = next((d for d in (",", "\t", ";", "|") if all(d in ln for ln in body)), None)
    if sep is None:
        sep = r"\s+" if any(len(ln.split()) > 1 for ln in body) else ","
    df = pd.read_csv(path, sep=sep, engine="python", skiprows=n_comment, header=0)
    if _header_row_is_data(df.columns):
        df = pd.read_csv(path, sep=sep, engine="python", skiprows=n_comment, header=None)
        df.columns = [f"col{i}" for i in range(df.shape[1])]
    df = df.dropna(axis=1, how="all")
    return _table_to_arrays(df), scalars, os.path.basename(path)


def _read_excel(path: str, sheet, interactive: bool):
    import pandas as pd
    engine = "xlrd" if path.lower().endswith(".xls") else None
    book = pd.read_excel(path, sheet_name=None, engine=engine)
    if sheet is not None:
        key = list(book)[int(sheet)] if str(sheet).isdigit() else sheet
        if key not in book:
            raise ValueError(f"sheet {sheet!r} not found; sheets: {list(book)}")
    else:
        # the sheet with the most numeric cells is the data; others are
        # typically metadata (and may carry a sampling-rate cell).
        score = {k: int(df.select_dtypes("number").count().sum()) for k, df in book.items()}
        key = max(score, key=score.get)
        plausible = [k for k, v in score.items() if v > 0.1 * score[key]]
        if len(plausible) > 1 and interactive:
            print("Sheets: " + ", ".join(f"[{i}] {k!r}" for i, k in enumerate(book)))
            key = list(book)[_prompt_index("Which sheet holds the ECG?", list(book).index(key), len(book))]
        print(f"Sheet: {key!r}" + (f" (of {len(book)})" if len(book) > 1 else ""))
    df = book[key]
    if _header_row_is_data(df.columns):
        df = pd.read_excel(path, sheet_name=key, header=None, engine=engine)
        df.columns = [f"col{i}" for i in range(df.shape[1])]
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    # metadata anywhere in the workbook: two-cell rows like ("fs", 1000)
    scalars = {}
    for other, odf in book.items():
        if other == key or odf.shape[1] < 1:
            continue
        for _, row in pd.concat([odf.columns.to_frame().T, odf.astype(object)]).iterrows():
            vals = [v for v in row.tolist() if not (isinstance(v, float) and np.isnan(v))]
            if len(vals) >= 2 and isinstance(vals[0], str):
                scalars[vals[0]] = vals[1]
    return _table_to_arrays(df), scalars, f"{os.path.basename(path)} [{key}]"


def _flatten(obj, prefix, arrays, scalars, labels):
    """Recursively collect 1-D numeric arrays, 2-D arrays (split per channel),
    scalars, and channel-label lists from nested dict/list/struct data."""
    if isinstance(obj, dict):
        if obj and all(isinstance(k, str) and k.lstrip("-").isdigit() for k in obj):
            obj = [obj[k] for k in sorted(obj, key=lambda s: int(s))]   # pandas orient="columns"
        else:
            if {"columns", "data"} <= set(obj) and isinstance(obj["data"], list):   # orient="split"
                import pandas as pd
                df = pd.DataFrame(obj["data"], columns=obj["columns"])
                for c in df.columns:
                    arrays[f"{prefix}{c}"] = df[c].to_numpy()
                return
            for k, v in obj.items():
                _flatten(v, f"{prefix}{k}" if not prefix else f"{prefix}.{k}", arrays, scalars, labels)
            return
    if isinstance(obj, (str, bytes)):
        return
    if isinstance(obj, (int, float, np.number)) and not isinstance(obj, bool):
        scalars[prefix or "value"] = obj
        return
    if isinstance(obj, list) and obj and all(isinstance(r, dict) for r in obj):   # records
        import pandas as pd
        df = pd.DataFrame(obj)
        for c in df.columns:
            arrays[f"{prefix}.{c}" if prefix else str(c)] = df[c].to_numpy()
        return
    if isinstance(obj, list) and obj and all(isinstance(s, str) for s in obj):
        name = _norm(prefix.split(".")[-1]) if prefix else ""
        if name in LABEL_NAMES:
            labels[prefix] = [str(s) for s in obj]
        elif len(obj) > 2:
            arrays[prefix or "values"] = np.asarray(obj)     # e.g. ISO timestamps
        return
    try:
        a = np.asarray(obj)
    except Exception:
        return
    if a.dtype == object:
        return
    a = a.squeeze()
    if a.ndim == 0:
        if _is_numeric(a):
            scalars[prefix or "value"] = a.item()
    elif a.ndim == 1 and len(a) > 1:
        arrays[prefix or "values"] = a
    elif a.ndim == 2:
        arrays[(prefix or "data", "2d")] = a


def _split_2d(arrays: dict, labels: dict) -> dict:
    """Split each 2-D array into per-channel arrays (the shorter axis is
    channels), naming them from a label list of the right length if any."""
    out = {}
    for k, a in arrays.items():
        if not (isinstance(k, tuple) and k[1] == "2d"):
            out[k] = a
            continue
        base = k[0]
        chans = a if a.shape[0] < a.shape[1] else a.T
        lab = next((v for v in labels.values() if len(v) == chans.shape[0]), None)
        for i in range(chans.shape[0]):
            out[f"{base}[{lab[i]}]" if lab else f"{base}[{i}]"] = chans[i]
    return out


def _read_json(path: str):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = [json.loads(line) for line in text.splitlines() if line.strip()]   # JSON Lines
    arrays, scalars, labels = {}, {}, {}
    _flatten(obj, "", arrays, scalars, labels)
    return _split_2d(arrays, labels), scalars, os.path.basename(path)


def _read_mat(path: str):
    arrays, scalars, labels = {}, {}, {}
    try:
        from scipy.io import loadmat
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
        backend = "scipy"

        def to_py(v):
            if hasattr(v, "_fieldnames"):
                return {f: to_py(getattr(v, f)) for f in v._fieldnames}
            if isinstance(v, np.ndarray) and v.dtype == object:
                items = [to_py(x) for x in v.ravel()]
                if all(isinstance(x, str) for x in items):
                    return items
                return items if len(items) != 1 else items[0]
            return v
        obj = {k: to_py(v) for k, v in data.items() if not k.startswith("__")}
    except NotImplementedError:
        import h5py
        backend = "h5py (MATLAB v7.3)"
        f = h5py.File(path, "r")

        def to_py(node):
            if isinstance(node, h5py.Group):
                return {k: to_py(node[k]) for k in node if not k.startswith("#")}
            v = node[()]
            if node.attrs.get("MATLAB_class", b"") in (b"char",):
                return "".join(chr(c) for c in np.asarray(v).ravel())
            if v.dtype == object:   # cell array of references
                return [to_py(f[r]) for r in np.asarray(v).ravel()]
            return np.asarray(v).T   # h5py reads MATLAB arrays transposed
        obj = {k: to_py(f[k]) for k in f if not k.startswith("#")}
    _flatten(obj, "", arrays, scalars, labels)
    if not arrays:
        raise ValueError(f"no numeric arrays found in {path} -- MATLAB `table`/`timetable` "
                         f"objects can't be read outside MATLAB; save the columns as plain "
                         f"variables (or a struct) instead")
    print(f"MAT file read with {backend}: {len(arrays)} array(s), "
          f"{len(scalars)} scalar(s)")
    return _split_2d(arrays, labels), scalars, os.path.basename(path)


# ══════════════════════════════════════════════════════════════════════════
#  Interactive prompt helpers
# ══════════════════════════════════════════════════════════════════════════

def _first(value, default=""):
    if isinstance(value, (list, tuple)) and len(value) > 0:
        return value[0]
    return value if value is not None else default


def _prompt_index(label: str, default, n: int, allow_none: bool = False):
    while True:
        suffix = "none" if default is None else default
        raw = input(f"{label} [{suffix}]: ").strip()
        if raw == "":
            return default
        if allow_none and raw.lower() in ("none", "n", "-"):
            return None
        try:
            idx = int(raw)
        except ValueError:
            print(f"  enter a number from 0 to {n - 1}" + (", or nothing/none" if allow_none else ""))
            continue
        if 0 <= idx < n:
            return idx
        print(f"  {idx} is out of range (0-{n - 1})")


# ══════════════════════════════════════════════════════════════════════════
#  XDF
# ══════════════════════════════════════════════════════════════════════════

def _xdf_labels(stream) -> list[str]:
    try:
        return [str(_first(c.get("label"), "")) for c in
                stream["info"]["desc"][0]["channels"][0]["channel"]]
    except (KeyError, IndexError, TypeError):
        return []


def _load_xdf(path: str, interactive: bool, channel=None) -> dict:
    import pyxdf
    streams, _ = pyxdf.load_xdf(path)
    if not streams:
        raise ValueError(f"no streams found in {path}")

    def describe(i, s):
        info = s["info"]
        return (f"[{i}] name={_first(info.get('name'), '?')!r} type={_first(info.get('type'), '?')!r} "
                f"channels={_first(info.get('channel_count'), '?')} "
                f"srate={_first(info.get('nominal_srate'), '0')} n_samples={len(s['time_series'])}")

    print(f"\nFound {len(streams)} stream(s) in {os.path.basename(path)}:")
    for i, s in enumerate(streams):
        print("  " + describe(i, s))

    stype = lambda s: str(_first(s["info"].get("type"), "")).lower()
    sname = lambda s: str(_first(s["info"].get("name"), "")).lower()
    has_data = lambda s: stype(s) != "markers" and len(s["time_stamps"]) > 10

    # Prefer a data stream with an ECG-labelled channel, then one named/typed
    # ECG, then any EEG-type stream (ECG piggybacked on an amplifier). Some
    # recorders write the same amp stream twice; the longest copy wins.
    by_label = [i for i, s in enumerate(streams) if has_data(s)
                and any(tok in lab.lower() for lab in _xdf_labels(s) for tok in ECG_NAME_TOKENS)]
    by_name = [i for i, s in enumerate(streams) if has_data(s)
               and any(tok in stype(s) + sname(s) for tok in ECG_NAME_TOKENS)]
    by_eeg = [i for i, s in enumerate(streams) if has_data(s) and stype(s) == "eeg"]
    pool = by_label or by_name or by_eeg or [i for i, s in enumerate(streams) if has_data(s)]
    if not pool:
        raise ValueError(f"no signal (non-marker) stream in {path}")
    ecg_idx = max(pool, key=lambda i: len(streams[i]["time_stamps"]))
    distinct = {(sname(streams[i]), len(_xdf_labels(streams[i]))) for i in pool}
    if interactive and len(distinct) > 1:
        ecg_idx = _prompt_index("Which stream is the ECG signal?", ecg_idx, len(streams))
    else:
        print(f"Using stream [{ecg_idx}] as ECG (auto-detected)")

    marker_candidates = [i for i, s in enumerate(streams)
                         if stype(s) == "markers" and len(s["time_stamps"]) >= 3]
    # Which entries of a Markers stream are stim onsets can't be inferred.
    # Exactly one non-empty stream: use it (cleaning falls back to
    # marker-free if it turns out not to be stim onsets).
    # Several: ask if interactive, otherwise use none.
    if len(marker_candidates) == 1:
        marker_idx = marker_candidates[0]
        print(f"Using stream [{marker_idx}] as stim markers (auto-detected; "
              f"marker-free detection is used instead if these don't line up with artifact)")
    elif len(marker_candidates) > 1 and interactive:
        marker_idx = _prompt_index("Which stream holds stim markers? (blank/none to skip)",
                                   None, len(streams), allow_none=True)
    elif len(marker_candidates) > 1:
        print(f"{len(marker_candidates)} candidate marker streams ({marker_candidates}) -- "
              f"ambiguous, so none used; marker-free detection will run")
        marker_idx = None
    else:
        marker_idx = None

    s = streams[ecg_idx]
    d = np.asarray(s["time_series"])
    ts = np.asarray(s["time_stamps"], dtype=np.float64)
    srate = float(_first(s["info"].get("nominal_srate"), 0) or 0)
    fs = srate if srate > 0 else 1.0 / float(np.median(np.diff(ts)))
    labels = _xdf_labels(s)

    if d.ndim == 1 or min(d.shape) == 1:
        sig, ch_desc = d.ravel().astype(np.float64), "single channel"
    else:
        transposed = d.shape[0] < d.shape[1]   # (channels, samples)
        n_ch = d.shape[0] if transposed else d.shape[1]
        ecg_named = [k for k, lab in enumerate(labels)
                     if any(tok in lab.lower() for tok in ECG_NAME_TOKENS)]
        if channel is not None:
            if isinstance(channel, int) or str(channel).isdigit():
                ch_idx = int(channel)
            else:
                ch_idx = [lab.lower() for lab in labels].index(str(channel).lower())
            ch_desc = "given"
        elif len(ecg_named) == 1:
            ch_idx, ch_desc = ecg_named[0], "channel label"
        else:
            ch_idx = _auto_pick_ecg_channel(d, transposed, n_ch, fs=fs)
            ch_desc = "most regular plausible heart rhythm in a 30 s excerpt"
            if interactive:
                ch_idx = _prompt_index("Which channel is ECG?", ch_idx, n_ch)
                ch_desc = "chosen interactively"
        sig = (d[ch_idx, :] if transposed else d[:, ch_idx]).astype(np.float64)
        lab = f" {labels[ch_idx]!r}" if ch_idx < len(labels) else ""
        ch_desc = f"channel {ch_idx}{lab} of {n_ch} ({ch_desc})"
    print(f"ECG: {ch_desc}")

    stim_times = None
    if marker_idx is not None:
        stim_times = np.asarray(streams[marker_idx]["time_stamps"], dtype=np.float64)
    return {"sig": sig, "ts": ts, "fs": fs, "stim_times": stim_times,
            "source": f"{os.path.basename(path)} :: stream {ecg_idx} :: {ch_desc}"}


def _auto_pick_ecg_channel(d: np.ndarray, transposed: bool, n_ch: int,
                           fs: float = 250.0) -> int:
    """Best guess at which channel of a multi-channel array is ECG.

    Tries to actually find heartbeats (NeuroKit2 `ecg_findpeaks`) on EVERY
    channel over a representative excerpt, and scores each by whether it
    has a plausible, REGULAR heart rate -- a non-ECG channel has nothing
    resembling a regular beat-to-beat rhythm even if it superficially
    "finds peaks."

    Channels with few distinct values (under 20% of samples) are skipped
    first: digital/status and coarsely quantized auxiliary channels can
    produce a spuriously regular "rhythm". Candidates are not pre-filtered
    by amplitude or kurtosis, since a noisy EEG channel with large
    transients can out-score a real ECG on those.
    """
    n_samp = d.shape[1] if transposed else d.shape[0]
    n_use = min(n_samp, int(30 * fs))
    start = max(0, (n_samp - n_use) // 2)  # a middle excerpt, not the start (settling artifacts)
    duration_s = n_use / fs
    min_unique = int(0.2 * n_use)

    try:
        import neurokit2 as nk
    except ImportError:
        nk = None

    best_ch, best_score = 0, -np.inf
    for ch in range(n_ch):
        x = (d[ch, start:start + n_use] if transposed else d[start:start + n_use, ch]).astype(np.float64)
        if np.std(x) == 0 or len(np.unique(x)) < min_unique:
            continue
        if nk is not None:
            try:
                info = nk.ecg_findpeaks(x, sampling_rate=fs)
                peaks = np.asarray(info["ECG_R_Peaks"])
            except Exception:
                continue
            if len(peaks) < 4:
                continue
            rr = np.diff(peaks) / fs
            hr = 60.0 / np.median(rr)
            if not (35 <= hr <= 200):
                continue
            regularity = 1.0 - min(1.0, float(np.std(rr) / np.mean(rr)))
            score = regularity * min(1.0, len(peaks) / (duration_s / 1.2))
        else:
            # No NeuroKit2 available -- fall back to plain kurtosis.
            xc = x - np.median(x)
            score = float(np.mean((xc / np.std(xc)) ** 4) - 3.0)
        if score > best_score:
            best_score, best_ch = score, ch
    return best_ch
