"""
Lab 3 — Step 4: check_schema_equity()
A reusable function that inspects a CSV dataset and prints warnings when:
  (a) A column is binary (yes/no, 0/1) AND represents something like
      income, background, region, or ability — a spectrum squished to 2 boxes.
  (b) A column has fewer than 4 categories AND represents a spectrum
      (income, region, ability, risk).

The function can be imported by other scripts or run standalone.
"""

import csv, os

# ── Sensitive keywords — columns whose names suggest a real-world spectrum ────
SENSITIVE_KEYWORDS = [
    "meal", "income", "economic", "need", "program", "risk",
    "region", "district", "background", "status", "ability",
    "support", "resource", "poverty", "socio", "free", "reduced",
]

def check_schema_equity(filepath_or_rows, label="dataset"):
    """
    Accepts either:
      - a file path string (CSV), or
      - a list of dicts (already loaded rows)

    Prints a warning report and returns the count of warnings found.
    """
    # Load if given a path
    if isinstance(filepath_or_rows, str):
        with open(filepath_or_rows, newline="") as f:
            rows = list(csv.DictReader(f))
        source = os.path.basename(filepath_or_rows)
    else:
        rows = filepath_or_rows
        source = label

    if not rows:
        print(f"  [{source}] No rows found.")
        return 0

    columns   = list(rows[0].keys())
    warnings  = []
    col_info  = []

    for col in columns:
        vals   = [r[col] for r in rows if r[col] != ""]
        unique = sorted(set(vals))
        n_uniq = len(unique)

        # Determine if numeric
        try:
            [float(v) for v in vals]
            is_numeric = True
        except ValueError:
            is_numeric = False

        # Is this column name sensitive?
        col_l     = col.lower()
        sensitive = any(k in col_l for k in SENSITIVE_KEYWORDS)

        # Rule (a): binary AND sensitive
        if n_uniq <= 2 and sensitive:
            msg = (f"Column '{col}' is BINARY ({unique}) but likely "
                   f"represents a spectrum (e.g. income/risk). "
                   f"Consider 4+ levels.")
            warnings.append(("BINARY_SENSITIVE", col, msg))

        # Rule (b): 2-3 categories AND sensitive AND not already numeric range
        elif 2 < n_uniq < 4 and sensitive and not is_numeric:
            msg = (f"Column '{col}' has only {n_uniq} categories {unique} "
                   f"but represents a spectrum. Consider expanding to 4+.")
            warnings.append(("FEW_CATEGORIES", col, msg))

        col_info.append((col, n_uniq, "NUMERIC" if is_numeric else "STRING",
                         sensitive, unique[:3]))

    # Print report
    print("=" * 72)
    print(f"  check_schema_equity — {source}")
    print("=" * 72)
    print(f"  Columns checked : {len(columns)}")
    print(f"  Warnings found  : {len(warnings)}")
    print()

    if warnings:
        print("  WARNINGS:")
        for kind, col, msg in warnings:
            print(f"  [{kind}] {msg}")
    else:
        print("  No equity warnings found. Schema looks well-structured.")

    print()
    print(f"  {'Column':<24} {'Unique':>6}  {'Type':<8}  {'Sensitive?':<11}  Sample")
    print("  " + "-"*68)
    for col, n_uniq, dtype, sensitive, sample in col_info:
        sens_str = "YES ⚠" if sensitive else "no"
        print(f"  {col:<24} {n_uniq:>6}  {dtype:<8}  {sens_str:<11}  "
              f"{str(sample)[:22]}")
    print("=" * 72)

    return len(warnings)


# ── Run standalone if executed directly ──────────────────────────────────────
if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))

    print()
    print("  Running check_schema_equity() — test on Lab 2 data")
    print()
    lab2_csv = os.path.join(base, "..", "Lab 2", "lab2_students.csv")
    n = check_schema_equity(lab2_csv, label="lab2_students.csv (ORIGINAL)")
    print(f"  → {n} warning(s) on original data\n")
