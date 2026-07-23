"""
Lab 2 — Step 1: Add District, Achievement Score, and Lat/Lon
Reads students_with_entropy.csv from Lab 1, adds:
  - district         : one of 8 made-up Georgia school district names
  - achievement_score: 0-100, correlated with meals_program to reflect
                       a real achievement gap (lower for high-need students)
  - lat / lon        : rough coordinates for each district (Georgia)

Output: lab2_students.csv  (saved to Lab 2 folder)
"""

import csv
import math
import os
import random

random.seed(42)

# ── District definitions ──────────────────────────────────────────────────────
# 8 fictional Georgia school districts with rough real-world coordinates
DISTRICTS = [
    {"name": "Westview USD",    "lat": 33.749,  "lon": -84.388},   # Atlanta area
    {"name": "Pinecrest ISD",   "lat": 32.083,  "lon": -81.099},   # Savannah area
    {"name": "Maplewood CSD",   "lat": 33.580,  "lon": -83.460},   # Madison area
    {"name": "Riverton USD",    "lat": 31.570,  "lon": -84.156},   # Albany area
    {"name": "Lakeshore ISD",   "lat": 34.296,  "lon": -83.824},   # Gainesville area
    {"name": "Oakdale USD",     "lat": 32.540,  "lon": -82.910},   # Dublin area
    {"name": "Cedarville CSD",  "lat": 30.835,  "lon": -83.978},   # Valdosta area
    {"name": "Highpoint ISD",   "lat": 34.014,  "lon": -84.616},   # Marietta area
]

# Achievement gap: meals_program="yes" students score lower on average
# This mirrors the real research finding from the Georgia paper
DISTRICT_BASE_SCORES = {
    "Westview USD":   72,
    "Pinecrest ISD":  65,
    "Maplewood CSD":  70,
    "Riverton USD":   58,   # lower-resource district
    "Lakeshore ISD":  75,
    "Oakdale USD":    63,
    "Cedarville CSD": 56,   # lowest-resource district
    "Highpoint ISD":  78,
}

def normal_sample(mu, sigma):
    """Box-Muller — no external libs."""
    while True:
        u1 = random.random() or 1e-10
        u2 = random.random()
        z  = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        v  = mu + sigma * z
        if 0 <= v <= 100:
            return round(v, 1)

# ── Load Lab 1 data ───────────────────────────────────────────────────────────
lab1_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "Lab 1", "students_with_entropy.csv"
)
lab2_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "lab2_students.csv"
)

rows = []
with open(lab1_path, newline="") as f:
    reader     = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        rows.append(row)

# ── Assign district and generate achievement score ────────────────────────────
for row in rows:
    district = random.choice(DISTRICTS)
    row["district"] = district["name"]
    row["lat"]      = district["lat"]
    row["lon"]      = district["lon"]

    base = DISTRICT_BASE_SCORES[district["name"]]
    # meals_program="yes" → subtract 8–14 points (achievement gap)
    gap  = random.uniform(8, 14) if row["meals_program"] == "yes" else 0
    row["achievement_score"] = normal_sample(base - gap, 8)

# ── Save ──────────────────────────────────────────────────────────────────────
new_fields = fieldnames + ["district", "achievement_score", "lat", "lon"]
with open(lab2_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=new_fields)
    writer.writeheader()
    writer.writerows(rows)

# ── Print first 10 rows ───────────────────────────────────────────────────────
print("=" * 78)
print("  Lab 2 — Step 1: Extended Dataset (first 10 rows)")
print("=" * 78)
print(f"  {'student_id':<12} {'district':<18} {'meals':<7} "
      f"{'at_risk':<9} {'ach_score':<11} {'lat':<9} {'lon'}")
print("-" * 78)
for r in rows[:10]:
    print(f"  {r['student_id']:<12} {r['district']:<18} "
          f"{r['meals_program']:<7} {r['at_risk']:<9} "
          f"{r['achievement_score']:<11} {r['lat']:<9} {r['lon']}")

# ── District averages ─────────────────────────────────────────────────────────
from collections import defaultdict
dist_scores = defaultdict(list)
for r in rows:
    dist_scores[r["district"]].append(float(r["achievement_score"]))

print()
print("=" * 78)
print("  Average Achievement Score by District")
print("=" * 78)
print(f"  {'District':<22} {'Students':>9} {'Avg Score':>11} {'Min':>7} {'Max':>7}")
print("-" * 78)
for d in sorted(dist_scores, key=lambda k: -sum(dist_scores[k])/len(dist_scores[k])):
    scores = dist_scores[d]
    avg = sum(scores) / len(scores)
    print(f"  {d:<22} {len(scores):>9} {avg:>11.1f} "
          f"{min(scores):>7.1f} {max(scores):>7.1f}")
print()
print(f"  Saved to: {lab2_path}")
print(f"  Total rows: {len(rows)}")
