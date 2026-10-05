"""Remove periodic electrical-stimulation artifact from ECG recordings.

    from stim_artifact_removal import clean_ecg, load_ecg_file
    out = clean_ecg(raw, ts, fs)          # stim markers optional

See README.md and docs/ for usage and the method.
"""
__version__ = "1.0.0"

from .core import (
    subtract_stim_template,
    default_pad_for_fs,
    despike_window,
    remove_stim_artifact,
    remove_stim_artifact_no_markers,
    calibrate_artifact_window,
    detect_pseudo_stim_times,
    default_curv_thresh_for_fs,
    resolve_curv_thresh,
    estimate_qrs_curvature,
    DEFAULT_CURV_THRESH,
    DEFAULT_PAD,
)
from .hrv import ecg_hrv, clean_and_hrv
from .preprocess import preprocess_ecg, decimate_to, detect_inverted
from .pipeline import clean_ecg
from .loaders import load_ecg_file

__all__ = [
    "clean_ecg",
    "subtract_stim_template",
    "default_pad_for_fs",
    "decimate_to",
    "detect_inverted",
    "load_ecg_file",
    "despike_window",
    "remove_stim_artifact",
    "remove_stim_artifact_no_markers",
    "calibrate_artifact_window",
    "detect_pseudo_stim_times",
    "default_curv_thresh_for_fs",
    "resolve_curv_thresh",
    "estimate_qrs_curvature",
    "ecg_hrv",
    "clean_and_hrv",
    "preprocess_ecg",
    "DEFAULT_CURV_THRESH",
    "DEFAULT_PAD",
]
