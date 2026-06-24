"""Automated tests for validate_data.py, run by pytest in the CI/CD pipeline."""
import sys
import os
# add the repo root (the folder above tests/) so Python can import validate_data
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from validate_data import check_missing, check_duplicates, check_target, check_ranges, validate

def sample_clean():
    return pd.DataFrame({
        "age": [30, 40, 50],
        "attendance_rate": [0.90, 0.95, 1.00],
        "performance_score": [70, 80, 90],
        "promoted": [0, 1, 0],
    })

def test_clean_data_passes():
    ok, _ = validate(sample_clean()); assert ok is True

def test_missing_is_detected():
    df = sample_clean(); df.loc[0, "age"] = np.nan
    ok, _ = check_missing(df); assert ok is False

def test_duplicates_are_detected():
    df = sample_clean(); df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    ok, _ = check_duplicates(df); assert ok is False

def test_out_of_range_is_detected():
    df = sample_clean(); df.loc[0, "attendance_rate"] = 1.5
    ok, _ = check_ranges(df); assert ok is False

def test_target_is_binary():
    ok, _ = check_target(sample_clean()); assert ok is True
