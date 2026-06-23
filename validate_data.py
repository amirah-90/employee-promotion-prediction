"""
Automated data validation for the Employee Promotion Prediction project.
Run: python validate_data.py data/sample_employees.csv
Checks missing values, duplicates, target column, and value ranges.
Exits 0 if all pass, 1 if any fail (used by CI/CD).
"""
import sys
import pandas as pd

TARGET = "promoted"
RANGE_RULES = {
    "age": (18, 70),
    "attendance_rate": (0, 1),
    "remote_work_ratio": (0, 1),
    "deadline_adherence_rate": (0, 1),
    "performance_score": (0, 100),
    "kpi_achievement_percent": (0, 100),
}

def check_missing(df):
    n = int(df.isnull().sum().sum())
    return n == 0, f"Missing cells: {n}"

def check_duplicates(df):
    n = int(df.duplicated().sum())
    return n == 0, f"Duplicate rows: {n}"

def check_target(df):
    if TARGET not in df.columns:
        return False, f"Target column '{TARGET}' is missing"
    values = set(int(v) for v in df[TARGET].dropna().unique())
    return values.issubset({0, 1}), f"Target '{TARGET}' values: {sorted(values)}"

def check_ranges(df):
    problems = {}
    for col, (low, high) in RANGE_RULES.items():
        if col in df.columns:
            bad = int(((df[col] < low) | (df[col] > high)).sum())
            if bad:
                problems[col] = bad
    return len(problems) == 0, f"Range violations: {problems if problems else 'none'}"

def validate(df):
    checks = [check_missing, check_duplicates, check_target, check_ranges]
    results, all_ok = [], True
    for chk in checks:
        ok, msg = chk(df)
        all_ok = all_ok and ok
        results.append((chk.__name__, "PASS" if ok else "FAIL", msg))
    return all_ok, results

def main(path):
    df = pd.read_csv(path)
    all_ok, results = validate(df)
    print(f"Validating: {path}  ({len(df)} rows)")
    print("-" * 55)
    for name, status, msg in results:
        print(f"[{status}] {name:16s} -> {msg}")
    print("-" * 55)
    print("All checks PASSED" if all_ok else "Some checks FAILED")
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_employees.csv"
    main(csv_path)
