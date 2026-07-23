"""
Lab 3 — Step 3: Redesign the Data Schema
Replaces oversimplified columns with richer alternatives:

  OLD → NEW
  meals_program (yes/no)  →  economic_need_level (4 levels: low/moderate/high/critical)
  district (single label) →  district (kept) + sub_region (adds within-district detail)

Saves the redesigned dataset as lab3_students_redesigned.csv
Prints a mapping table showing old → new column values.
"""

import csv, os, random, math
random.seed(42)

base     = os.path.dirname(os.path.abspath(__file__))
lab2_csv = os.path.join(base, "..", "Lab 2", "lab2_students.csv")
out_csv  = os.path.join(base, "lab3_students_redesigned.csv")

rows = list(csv.DictReader(open(lab2_csv)))

# ── Sub-region mapping (2-3 sub-regions per district) ────────────────────────
SUB_REGIONS = {
    "Westview USD":   ["Westview North", "Westview South", "Westview Central"],
    "Pinecrest ISD":  ["Pinecrest East",  "Pinecrest West"],
    "Maplewood CSD":  ["Maplewood Urban", "Maplewood Rural"],
    "Riverton USD":   ["Riverton Core",   "Riverton Outskirts"],
    "Lakeshore ISD":  ["Lakeshore Hills", "Lakeshore Valley", "Lakeshore Bay"],
    "Oakdale USD":    ["Oakdale Central", "Oakdale Suburbs"],
    "Cedarville CSD": ["Cedarville Town", "Cedarville County"],
    "Highpoint ISD":  ["Highpoint Ridge", "Highpoint Glen",  "Highpoint Park"],
}

# ── economic_need_level mapping ───────────────────────────────────────────────
# old meals_program yes/no → new 4-level scale based on achievement_score proxy
# low need (high scores) → moderate → high → critical (lowest scores)
def economic_need_level(row):
    """
    Derive a 4-level economic need score.
    meals_program=no  → low or moderate (weighted toward low)
    meals_program=yes → high or critical (weighted by achievement_score)
    """
    meals = row["meals_program"] == "yes"
    ach   = float(row["achievement_score"])

    if not meals:
        # Non-meals students: low or moderate need
        return "low" if ach >= 65 else "moderate"
    else:
        # Meals students: high or critical need
        return "critical" if ach < 55 else "high"

# ── Apply redesign ─────────────────────────────────────────────────────────────
mapping_examples = []   # collect a few rows for the mapping table

for row in rows:
    old_meals   = row["meals_program"]
    new_level   = economic_need_level(row)
    district    = row["district"]
    sub_region  = random.choice(SUB_REGIONS.get(district, ["Unknown"]))

    row["economic_need_level"] = new_level
    row["sub_region"]          = sub_region

    if len(mapping_examples) < 12:
        mapping_examples.append({
            "student_id":         row["student_id"],
            "old_meals_program":  old_meals,
            "new_economic_level": new_level,
            "old_district":       district,
            "new_sub_region":     sub_region,
        })

# ── Save redesigned CSV ───────────────────────────────────────────────────────
old_fields = list(rows[0].keys())
# Insert new fields after their old counterparts; remove meals_program
new_fields = []
for f in old_fields:
    if f == "meals_program":
        new_fields.append("economic_need_level")   # replace
    elif f == "district":
        new_fields.extend(["district", "sub_region"])  # extend
    else:
        new_fields.append(f)

with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=new_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

# ── Print mapping table ───────────────────────────────────────────────────────
print("=" * 78)
print("  Lab 3 — Step 3: Schema Redesign")
print("=" * 78)
print()
print("  OLD COLUMN          NEW COLUMN             Rationale")
print("  " + "-"*72)
print("  meals_program       economic_need_level     4 levels instead of 2;")
print("  (yes / no)          (low/moderate/          captures the spectrum of")
print("                       high/critical)          income need more accurately")
print()
print("  district            district + sub_region   Adds within-district detail")
print("  (single label)      (district kept;         so communities inside the")
print("                       sub_region added)       same district aren't lumped")
print()
print("  " + "="*72)
print("  Sample mapping (first 12 students):")
print()
print(f"  {'student_id':<12} {'old meals':<12} {'new level':<12} "
      f"{'old district':<18} {'new sub_region'}")
print("  " + "-"*72)
for ex in mapping_examples:
    print(f"  {ex['student_id']:<12} {ex['old_meals_program']:<12} "
          f"{ex['new_economic_level']:<12} {ex['old_district']:<18} "
          f"{ex['new_sub_region']}")

# Distribution of new economic_need_level
from collections import Counter
level_counts = Counter(r["economic_need_level"] for r in rows)
print()
print("  Distribution of economic_need_level:")
for level in ["low", "moderate", "high", "critical"]:
    n   = level_counts[level]
    bar = "█" * int(n / len(rows) * 40)
    print(f"    {level:<10} {n:>4} ({n/len(rows)*100:.1f}%)  {bar}")

print()
print(f"  Redesigned data saved to: {out_csv}")
print(f"  Columns: {', '.join(new_fields)}")
