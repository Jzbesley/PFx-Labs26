"""
Lab 2 — Step 2: Feature Ranking with Mutual Information
Ranks numeric and binary columns by how useful they are for predicting at_risk.

Mutual information (MI) measures how much knowing a feature reduces uncertainty
about the target.  Higher MI = more useful column.

We compute a discretized MI estimate: bin each numeric column, then count
joint frequencies against the binary target.

Output: prints ranked table + plain-English explanation of the top column.
"""

import csv
import math
import os
from collections import defaultdict

# ── Load data ─────────────────────────────────────────────────────────────────
lab2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab2_students.csv")

rows = []
with open(lab2_path, newline="") as f:
    for row in csv.DictReader(f):
        rows.append(row)

# Target: at_risk (binary 0/1)
y = [1 if r["at_risk"] == "yes" else 0 for r in rows]

# ── Feature definitions ───────────────────────────────────────────────────────
# (name, extractor, n_bins)  — use n_bins=2 for already-binary columns
FEATURES = [
    ("attendance_rate",    lambda r: float(r["attendance_rate"])    if r["attendance_rate"]    != "" else 0.5, 10),
    ("avg_quiz_score",     lambda r: float(r["avg_quiz_score"])     if r["avg_quiz_score"]     != "" else 50,  10),
    ("attendance_entropy", lambda r: float(r["attendance_entropy"]) if r["attendance_entropy"] != "" else 0.5, 10),
    ("achievement_score",  lambda r: float(r["achievement_score"])  if r["achievement_score"]  != "" else 50,  10),
    ("meals_program",      lambda r: 1 if r["meals_program"] == "yes" else 0,                                   2),
]

def discretize(values, n_bins):
    """Bin a list of floats into n_bins equal-width integer buckets."""
    lo, hi = min(values), max(values)
    span   = (hi - lo) or 1e-8
    bins   = []
    for v in values:
        b = int((v - lo) / span * n_bins)
        bins.append(min(b, n_bins - 1))   # clamp top value into last bin
    return bins

def mutual_information(x_bins, y):
    """
    MI(X; Y) = Σ_x Σ_y P(x,y) * log( P(x,y) / (P(x)*P(y)) )
    """
    n = len(y)
    # Joint counts P(x, y)
    joint  = defaultdict(int)
    count_x = defaultdict(int)
    count_y = defaultdict(int)
    for xi, yi in zip(x_bins, y):
        joint[(xi, yi)] += 1
        count_x[xi]     += 1
        count_y[yi]      += 1

    mi = 0.0
    for (xi, yi), cnt in joint.items():
        pxy = cnt / n
        px  = count_x[xi] / n
        py  = count_y[yi] / n
        if pxy > 0 and px > 0 and py > 0:
            mi += pxy * math.log(pxy / (px * py))
    return max(0.0, mi)

# ── Compute MI for each feature ───────────────────────────────────────────────
results = []
for name, extractor, n_bins in FEATURES:
    raw   = [extractor(r) for r in rows]
    bins  = discretize(raw, n_bins)
    mi    = mutual_information(bins, y)
    results.append((name, mi))

results.sort(key=lambda x: -x[1])

# ── Print ranked table ────────────────────────────────────────────────────────
max_mi = results[0][1] if results[0][1] > 0 else 1

print("=" * 62)
print("  Lab 2 — Step 2: Feature Ranking (Mutual Information)")
print("=" * 62)
print(f"  Target column: at_risk")
print(f"  {'Rank':<6} {'Feature':<22} {'MI Score':>10}   Bar")
print("-" * 62)
for rank, (name, mi) in enumerate(results, 1):
    bar_len = int(mi / max_mi * 25)
    bar     = "█" * bar_len
    print(f"  {rank:<6} {name:<22} {mi:>10.4f}   {bar}")

print("=" * 62)

top_name, top_mi = results[0]
print()
print(f"  Top feature: {top_name}  (MI = {top_mi:.4f})")
print()

explanations = {
    "attendance_rate": (
        "attendance_rate tells us the most about whether a student is at risk.\n"
        "  This makes sense: a student who misses a lot of school is more likely\n"
        "  to fall behind and get flagged as needing extra help. Knowing a student's\n"
        "  attendance rate cuts our uncertainty about their risk status more than\n"
        "  any other single piece of information we have."
    ),
    "avg_quiz_score": (
        "avg_quiz_score tells us the most about whether a student is at risk.\n"
        "  Students with consistently low quiz scores are more likely to need extra\n"
        "  support. Quiz scores directly reflect academic performance, so they carry\n"
        "  strong predictive power for identifying at-risk students."
    ),
    "achievement_score": (
        "achievement_score tells us the most about whether a student is at risk.\n"
        "  This reflects overall academic standing in the district. Since it is\n"
        "  designed to correlate with income and school resources, it captures\n"
        "  systemic gaps that other columns miss individually."
    ),
    "meals_program": (
        "meals_program tells us the most about whether a student is at risk.\n"
        "  This is a proxy for family income — students who qualify for free/reduced\n"
        "  meals are more likely to face resource challenges that affect their\n"
        "  academic performance, making this a strong (though potentially unfair)\n"
        "  predictor of risk."
    ),
    "attendance_entropy": (
        "attendance_entropy tells us the most about whether a student is at risk.\n"
        "  High entropy means a student's attendance jumps around unpredictably\n"
        "  across grading periods. That instability is a warning sign that something\n"
        "  is disrupting the student's ability to show up consistently."
    ),
}

print(f"  Plain-English explanation:")
print(f"  {explanations.get(top_name, top_name + ' is the most informative feature.')}")
print()

# Save rankings to a simple CSV for use in later steps
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_rankings.csv")
with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rank", "feature", "mi_score"])
    for rank, (name, mi) in enumerate(results, 1):
        w.writerow([rank, name, round(mi, 4)])
print(f"  Rankings saved to: {out_path}")
