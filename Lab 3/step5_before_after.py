"""
Lab 3 — Step 5: Before vs. After Comparison
Runs check_schema_equity() on:
  1. The original Lab 2 dataset  (lab2_students.csv)
  2. The redesigned dataset      (lab3_students_redesigned.csv)
Prints a comparison table showing how many warnings dropped.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step4_equity_checker import check_schema_equity

base         = os.path.dirname(os.path.abspath(__file__))
original_csv = os.path.join(base, "..", "Lab 2", "lab2_students.csv")
redesign_csv = os.path.join(base, "lab3_students_redesigned.csv")

print()
print("=" * 72)
print("  Lab 3 — Step 5: Before vs. After Schema Redesign")
print("=" * 72)
print()

print("  ── BEFORE (original Lab 2 data) ──")
n_before = check_schema_equity(original_csv, label="lab2_students.csv")

print()
print("  ── AFTER (redesigned Lab 3 data) ──")
n_after = check_schema_equity(redesign_csv, label="lab3_students_redesigned.csv")

print()
print("=" * 72)
print("  SUMMARY")
print("=" * 72)
print(f"  Warnings BEFORE redesign : {n_before}")
print(f"  Warnings AFTER  redesign : {n_after}")
print(f"  Warnings removed         : {n_before - n_after}")
print()
if n_before > n_after:
    print("  The redesign reduced equity warnings by replacing the binary")
    print("  'meals_program' column with a 4-level 'economic_need_level',")
    print("  and adding 'sub_region' to capture within-district differences.")
elif n_before == n_after:
    print("  Warning count unchanged — check column names and sensitive keywords.")
else:
    print("  Warning count increased — review newly added columns.")
print("=" * 72)
