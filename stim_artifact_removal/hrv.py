"""HRV helpers built on NeuroKit2 (kept separate so ``core`` does not
require NeuroKit2)."""
from __future__ import annotations

import neurokit2 as nk
import numpy as np

from .core import remove_stim_artifact


def ecg_hrv(sig: np.ndarray, fs: float) -> dict:
    """NeuroKit2 ``ecg_process`` + ``hrv_time`` on ``sig`` as given.

    Returns ``{n_peaks, MeanNN, SDNN, RMSSD, pNN50}``, or ``{"error": msg}``
    if NeuroKit2 fails (e.g. too few beats). Never raises.
    """
    try:
        _, info = nk.ecg_process(sig, sampling_rate=fs)
        hrv = nk.hrv_time(info["ECG_R_Peaks"], sampling_rate=fs, show=False)
        return {
            "n_peaks": len(info["ECG_R_Peaks"]),
            "MeanNN": float(hrv["HRV_MeanNN"].iloc[0]),
            "SDNN": float(hrv["HRV_SDNN"].iloc[0]),
            "RMSSD": float(hrv["HRV_RMSSD"].iloc[0]),
            "pNN50": float(hrv["HRV_pNN50"].iloc[0]),
        }
    except Exception as e:
        return {"error": str(e)}


def clean_and_hrv(sig: np.ndarray, ts: np.ndarray, stim_times: np.ndarray,
                  fs: float, **despike_kwargs) -> tuple[np.ndarray, dict, dict]:
    """``remove_stim_artifact`` followed by ``ecg_hrv``.

    Returns ``(cleaned_signal, diagnostics, hrv)``.
    """
    cleaned, diag = remove_stim_artifact(sig, ts, stim_times, fs, **despike_kwargs)
    hrv = ecg_hrv(cleaned, fs)
    return cleaned, diag, hrv
