"""
Lab 3 — Step 1: Check Data for Hidden Bias
Reads the Lab 2 extended dataset and classifies every column into:
  - BINARY        : yes/no or 0/1 (2 unique values)
  - FEW_CATEGORIES: 3-9 unique string/int values
  - NUMERIC_RANGE : continuous float/int with many unique values
  - ID            : identifier columns

Flags any column that squeezes a real-life spectrum into too few boxes.
"""

import csv, os

base     = os.path.dirname(os.path.abspath(__file__))
lab2_csv = os.path.join(base, "..", "Lab 2", "lab2_students.csv")

rows = list(csv.DictReader(open(lab2_csv)))
columns = list(rows[0].keys())

# Classify each column
def classify(col, rows):
    vals = [r[col] for r in rows if r[col] != ""]
    unique = set(vals)
    n_unique = len(unique)

    # Try numeric
    try:
        nums = [float(v) for v in vals]
        if n_unique <= 2:
            return "BINARY", n_unique, sorted(unique)
        elif n_unique <= 9:
            return "FEW_CATEGORIES", n_unique, sorted(unique)
        else:
            lo, hi = min(nums), max(nums)
            return "NUMERIC_RANGE", n_unique, (round(lo,2), round(hi,2))
    except ValueError:
        # String column
        if n_unique <= 2:
            return "BINARY", n_unique, sorted(unique)
        elif n_unique <= 9:
            return "FEW_CATEGORIES", n_unique, sorted(unique)
        else:
            return "MANY_STRINGS", n_unique, list(sorted(unique))[:3]

# Sensitive columns that might be oversimplified
SENSITIVE_KEYWORDS = ["meal", "income", "program", "risk", "economic",
                      "region", "district", "background", "status"]

def is_sensitive(col):
    col_l = col.lower()
    return any(k in col_l for k in SENSITIVE_KEYWORDS)

results = []
for col in columns:
    kind, n_unique, sample = classify(col, rows)
    flag = ""
    if kind == "BINARY" and is_sensitive(col):
        flag = "⚠ BINARY — squeezes a spectrum into just 2 boxes"
    elif kind == "FEW_CATEGORIES" and is_sensitive(col) and n_unique < 4:
        flag = "⚠ FEW_CATEGORIES — may hide real differences"
    results.append((col, kind, n_unique, sample, flag))

# Print table
print("=" * 80)
print("  Lab 3 — Step 1: Column Bias Classification")
print("=" * 80)
print(f"  {'Column':<22} {'Type':<16} {'Unique':>6}   Sample Values")
print("-" * 80)
for col, kind, n_unique, sample, flag in results:
    sample_str = str(sample)[:30]
    print(f"  {col:<22} {kind:<16} {n_unique:>6}   {sample_str}")
    if flag:
        print(f"  {'':22} {flag}")
print("=" * 80)
print()
print("  FLAGGED COLUMNS:")
flagged = [(col, flag) for col, _, _, _, flag in results if flag]
if flagged:
    for col, flag in flagged:
        print(f"  • {col}: {flag}")
else:
    print("  None found.")
print()
print("  KEY INSIGHT:")
print("  'meals_program' is binary (yes/no) but income need is a spectrum.")
print("  'district' lumps many communities into one label, hiding sub-group gaps.")
print("  These will be redesigned in Step 3.")
