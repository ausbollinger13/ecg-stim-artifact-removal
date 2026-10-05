import json

import numpy as np
import pandas as pd
import pytest

from stim_artifact_removal import load_ecg_file

FS = 500.0


@pytest.fixture(scope="module")
def signal():
    from conftest import synthetic_ecg
    return synthetic_ecg(fs=FS, duration=20.0)


def check(rec, signal):
    assert rec["fs"] == pytest.approx(FS)
    assert len(rec["sig"]) == len(signal)
    np.testing.assert_allclose(rec["sig"], signal, rtol=1e-6, atol=1e-3)
    np.testing.assert_allclose(np.diff(rec["ts"]), 1.0 / FS, rtol=1e-6)


def test_csv_with_time_column(tmp_path, signal):
    p = tmp_path / "rec.csv"
    pd.DataFrame({"time": np.arange(len(signal)) / FS, "ECG": signal}).to_csv(p, index=False)
    check(load_ecg_file(str(p), interactive=False), signal)


def test_csv_time_in_ms(tmp_path, signal):
    p = tmp_path / "rec.csv"
    pd.DataFrame({"time_ms": np.arange(len(signal)) * 1000.0 / FS,
                  "ecg": signal}).to_csv(p, index=False)
    check(load_ecg_file(str(p), interactive=False), signal)


def test_mat(tmp_path, signal):
    from scipy.io import savemat
    p = tmp_path / "rec.mat"
    savemat(p, {"ecg": signal, "fs": FS})
    check(load_ecg_file(str(p), interactive=False), signal)


def test_json_columns(tmp_path, signal):
    p = tmp_path / "rec.json"
    p.write_text(json.dumps({"fs": FS, "ecg": signal.tolist()}))
    check(load_ecg_file(str(p), interactive=False), signal)


def test_excel(tmp_path, signal):
    pytest.importorskip("openpyxl")
    p = tmp_path / "rec.xlsx"
    pd.DataFrame({"time": np.arange(len(signal)) / FS, "ECG": signal}).to_excel(p, index=False)
    check(load_ecg_file(str(p), interactive=False), signal)


def test_explicit_fs_without_time(tmp_path, signal):
    p = tmp_path / "rec.txt"
    np.savetxt(p, signal)
    check(load_ecg_file(str(p), fs=FS, interactive=False), signal)
