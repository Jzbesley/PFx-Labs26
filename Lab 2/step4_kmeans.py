"""
Lab 2 — Step 4: K-Means Clustering
Groups students into 4 clusters based on numeric features.
Uses K-Means++ initialization for stable, well-separated clusters.
Reports average achievement_score per cluster and saves cluster labels.
"""

import csv, math, os, random
random.seed(42)

# ── Load data ─────────────────────────────────────────────────────────────────
base = os.path.dirname(os.path.abspath(__file__))
rows = []
with open(os.path.join(base, "lab2_students.csv"), newline="") as f:
    for r in csv.DictReader(f):
        att = r["attendance_rate"]
        if att == "": continue
        rows.append({
            "student_id":         r["student_id"],
            "attendance_rate":    float(att),
            "avg_quiz_score":     float(r["avg_quiz_score"]) if r["avg_quiz_score"] != "" else 50.0,
            "attendance_entropy": float(r["attendance_entropy"]),
            "meals_program":      1.0 if r["meals_program"] == "yes" else 0.0,
            "achievement_score":  float(r["achievement_score"]),
            "at_risk":            r["at_risk"],
            "district":           r["district"],
        })

# ── Standardize features ──────────────────────────────────────────────────────
FEAT_KEYS = ["attendance_rate", "avg_quiz_score", "attendance_entropy",
             "meals_program", "achievement_score"]

def col_mean_sd(data, key):
    vals = [d[key] for d in data]
    mu   = sum(vals) / len(vals)
    sd   = math.sqrt(sum((v-mu)**2 for v in vals) / len(vals)) or 1e-8
    return mu, sd

stats = {k: col_mean_sd(rows, k) for k in FEAT_KEYS}

def std_vec(row):
    return [(row[k] - stats[k][0]) / stats[k][1] for k in FEAT_KEYS]

X = [std_vec(r) for r in rows]
n, d = len(X), len(X[0])

# ── K-Means++ initialization ──────────────────────────────────────────────────
K = 4

def dist2(a, b):
    return sum((ai - bi)**2 for ai, bi in zip(a, b))

def kmeanspp_init(X, k):
    centers = [random.choice(X)]
    for _ in range(k - 1):
        dists   = [min(dist2(x, c) for c in centers) for x in X]
        total   = sum(dists)
        r       = random.uniform(0, total)
        running = 0
        for x, d in zip(X, dists):
            running += d
            if running >= r:
                centers.append(x)
                break
        else:
            centers.append(X[-1])
    return [list(c) for c in centers]

centers = kmeanspp_init(X, K)

# ── K-Means iterations ────────────────────────────────────────────────────────
def assign(X, centers):
    return [min(range(K), key=lambda k: dist2(x, centers[k])) for x in X]

def update_centers(X, labels, k):
    new_centers = []
    for ki in range(k):
        members = [X[i] for i in range(len(X)) if labels[i] == ki]
        if not members:
            new_centers.append(centers[ki])   # keep old center if empty
        else:
            new_centers.append([sum(m[j] for m in members) / len(members)
                                 for j in range(d)])
    return new_centers

for _ in range(100):
    labels      = assign(X, centers)
    new_centers = update_centers(X, labels, K)
    if new_centers == centers:
        break
    centers = new_centers

# ── Attach cluster labels to rows ─────────────────────────────────────────────
for row, label in zip(rows, labels):
    row["cluster"] = label

# ── Compute per-cluster stats ─────────────────────────────────────────────────
from collections import defaultdict
cluster_data = defaultdict(list)
for row in rows:
    cluster_data[row["cluster"]].append(row)

# Sort clusters by average achievement_score descending for readability
def cluster_avg_ach(k):
    return sum(r["achievement_score"] for r in cluster_data[k]) / len(cluster_data[k])

sorted_clusters = sorted(cluster_data.keys(), key=cluster_avg_ach, reverse=True)

print("=" * 72)
print("  Lab 2 — Step 4: K-Means Clustering (K=4)")
print("=" * 72)
print(f"  Features used: {', '.join(FEAT_KEYS)}")
print()
print(f"  {'Cluster':<10} {'Size':>6} {'Avg Ach':>9} {'Avg Att':>9} "
      f"{'Avg Quiz':>10} {'At-Risk %':>11} {'Meals %':>9}")
print("-" * 72)

for k in sorted_clusters:
    members = cluster_data[k]
    n_k     = len(members)
    avg_ach  = sum(r["achievement_score"]  for r in members) / n_k
    avg_att  = sum(r["attendance_rate"]    for r in members) / n_k
    avg_quiz = sum(r["avg_quiz_score"]     for r in members) / n_k
    pct_risk = sum(1 for r in members if r["at_risk"] == "yes") / n_k * 100
    pct_meal = sum(r["meals_program"]      for r in members) / n_k * 100
    print(f"  Cluster {k:<3} {n_k:>6} {avg_ach:>9.1f} {avg_att:>9.3f} "
          f"{avg_quiz:>10.1f} {pct_risk:>10.1f}% {pct_meal:>8.1f}%")

print()
print("  Interpretation:")
highest = sorted_clusters[0]
lowest  = sorted_clusters[-1]
hm = cluster_data[highest]
lm = cluster_data[lowest]
print(f"  Cluster {highest} has the HIGHEST avg achievement ({cluster_avg_ach(highest):.1f}) "
      f"— {sum(1 for r in hm if r['at_risk']=='yes')/len(hm)*100:.0f}% at-risk, "
      f"{sum(r['meals_program'] for r in hm)/len(hm)*100:.0f}% on meals program.")
print(f"  Cluster {lowest} has the LOWEST  avg achievement ({cluster_avg_ach(lowest):.1f}) "
      f"— {sum(1 for r in lm if r['at_risk']=='yes')/len(lm)*100:.0f}% at-risk, "
      f"{sum(r['meals_program'] for r in lm)/len(lm)*100:.0f}% on meals program.")

# ── Save cluster-labeled data ─────────────────────────────────────────────────
out_path = os.path.join(base, "lab2_students_clustered.csv")
fieldnames = list(rows[0].keys())
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print()
print(f"  Clustered data saved to: {out_path}")
print("=" * 72)
