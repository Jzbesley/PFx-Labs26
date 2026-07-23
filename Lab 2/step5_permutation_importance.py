"""
Lab 2 — Step 5: Permutation Importance
Loads the trained MLP weights from Step 3, then for each feature:
  1. Scrambles (permutes) that column across all test rows
  2. Measures how much worse the model's MSE gets
  3. The bigger the increase in MSE → the more important the feature

Produces a text-based bar chart (no external libs needed).
Also saves an HTML bar chart to permutation_importance.html.
"""

import csv, math, os, random
random.seed(99)   # different seed so permutations are independent

base = os.path.dirname(os.path.abspath(__file__))

# ── Reload MLP weights ────────────────────────────────────────────────────────
W1 = [[0.0]*4  for _ in range(16)]
b1 = [0.0]*16
W2 = [[0.0]*16 for _ in range(8)]
b2 = [0.0]*8
W3 = [[0.0]*8]
b3 = [0.0]

with open(os.path.join(base, "mlp_weights.csv"), newline="") as f:
    for row in csv.DictReader(f):
        layer = row["layer"]
        r_idx, c_idx, val = int(row["row"]), int(row["col"]), float(row["value"])
        if layer == "W1": W1[r_idx][c_idx] = val
        elif layer == "b1": b1[r_idx] = val
        elif layer == "W2": W2[r_idx][c_idx] = val
        elif layer == "b2": b2[r_idx] = val
        elif layer == "W3": W3[0][c_idx] = val
        elif layer == "b3": b3[0] = val

# ── Reload standardization params ────────────────────────────────────────────
FEAT_KEYS = ["attendance_rate", "avg_quiz_score", "attendance_entropy", "meals_program"]
feat_stats = {}
tgt_mu = tgt_sd = None

with open(os.path.join(base, "mlp_params.csv"), newline="") as f:
    for row in csv.DictReader(f):
        if row["key"] == "target_mean":
            tgt_mu = float(row["mean"])
            tgt_sd = float(row["sd"])
        else:
            feat_stats[row["key"]] = (float(row["mean"]), float(row["sd"]))

def std_feat(val, key):
    mu, sd = feat_stats[key]
    return (val - mu) / sd

def inv_target(v):
    return v * tgt_sd + tgt_mu

def relu(x): return max(0.0, x)

def predict(x_std):
    z1 = [sum(W1[j][i]*x_std[i] for i in range(4))  + b1[j] for j in range(16)]
    a1 = [relu(v) for v in z1]
    z2 = [sum(W2[j][i]*a1[i]   for i in range(16)) + b2[j] for j in range(8)]
    a2 = [relu(v) for v in z2]
    return inv_target(sum(W3[0][i]*a2[i] for i in range(8)) + b3[0])

# ── Load test data (last 20% — same split logic as Step 3) ───────────────────
rows_all = []
with open(os.path.join(base, "lab2_students.csv"), newline="") as f:
    for r in csv.DictReader(f):
        att = r["attendance_rate"]
        if att == "": continue
        rows_all.append({
            "attendance_rate":    float(att),
            "avg_quiz_score":     float(r["avg_quiz_score"]) if r["avg_quiz_score"] != "" else 50.0,
            "attendance_entropy": float(r["attendance_entropy"]),
            "meals_program":      1.0 if r["meals_program"] == "yes" else 0.0,
            "achievement_score":  float(r["achievement_score"]),
        })

random.seed(42)          # must match Step 3 to get the same shuffle
random.shuffle(rows_all)
split     = int(0.8 * len(rows_all))
test_rows = rows_all[split:]
random.seed(99)          # back to permutation seed

X_test = [[std_feat(r[k], k) for k in FEAT_KEYS] for r in test_rows]
y_test = [r["achievement_score"] for r in test_rows]

def mse(preds, actuals):
    return sum((p - a)**2 for p, a in zip(preds, actuals)) / len(actuals)

# Baseline MSE
baseline_preds = [predict(x) for x in X_test]
baseline_mse   = mse(baseline_preds, y_test)

# ── Permutation importance ────────────────────────────────────────────────────
N_REPEATS = 10   # scramble each column 10 times, average the result
importances = {}

for fi, feat in enumerate(FEAT_KEYS):
    deltas = []
    for _ in range(N_REPEATS):
        # Scramble feature fi
        col_vals = [x[fi] for x in X_test]
        random.shuffle(col_vals)
        X_perm   = [x[:fi] + [col_vals[i]] + x[fi+1:] for i, x in enumerate(X_test)]
        perm_mse = mse([predict(x) for x in X_perm], y_test)
        deltas.append(perm_mse - baseline_mse)
    importances[feat] = sum(deltas) / len(deltas)

# Sort descending
sorted_imp = sorted(importances.items(), key=lambda x: -x[1])
max_imp    = max(v for _, v in sorted_imp) or 1

# ── Text bar chart ────────────────────────────────────────────────────────────
print("=" * 68)
print("  Lab 2 — Step 5: Permutation Importance")
print("=" * 68)
print(f"  Baseline MSE: {baseline_mse:.2f}")
print(f"  Method: scramble each column {N_REPEATS}x, average MSE increase")
print()
print(f"  {'Feature':<24} {'MSE increase':>13}   Bar (importance)")
print("-" * 68)
for feat, imp in sorted_imp:
    bar = "█" * int(imp / max_imp * 30)
    print(f"  {feat:<24} {imp:>13.2f}   {bar}")
print("=" * 68)
print()
print("  The taller the bar, the more the model relied on that feature.")
print("  Scrambling an important column breaks the model badly (big MSE jump).")
print("  A tiny MSE increase means the model barely used that column.")

# ── HTML bar chart ────────────────────────────────────────────────────────────
features = [f for f, _ in sorted_imp]
values   = [round(v, 3) for _, v in sorted_imp]

# Color scale: highest = deep red, lowest = steel blue
colors = ["#c0392b", "#e67e22", "#f1c40f", "#2980b9"][:len(features)]
while len(colors) < len(features):
    colors.append("#95a5a6")

bars_html = ""
for feat, val, color in zip(features, values, colors):
    width = int(val / max_imp * 400)
    bars_html += f"""
        <div class="bar-row">
          <div class="label">{feat}</div>
          <div class="bar" style="width:{width}px; background:{color};">
            <span class="val">{val:.3f}</span>
          </div>
        </div>"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Lab 2 — Permutation Importance</title>
  <style>
    body       {{ font-family: Arial, sans-serif; max-width: 700px;
                  margin: 40px auto; background: #f9f9f9; color: #222; }}
    h2         {{ color: #2c3e50; }}
    .subtitle  {{ color: #555; font-size: 0.9em; margin-bottom: 24px; }}
    .bar-row   {{ display: flex; align-items: center; margin: 10px 0; }}
    .label     {{ width: 200px; font-size: 0.92em; text-align: right;
                  padding-right: 12px; color: #333; }}
    .bar       {{ height: 32px; border-radius: 4px; display: flex;
                  align-items: center; min-width: 4px; transition: width 0.3s; }}
    .val       {{ color: white; font-weight: bold; font-size: 0.85em;
                  padding-left: 8px; white-space: nowrap; }}
    .footer    {{ margin-top: 32px; font-size: 0.82em; color: #888; }}
    table      {{ border-collapse: collapse; width: 100%; margin-top: 24px; }}
    th, td     {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
    th         {{ background: #2c3e50; color: white; }}
    tr:nth-child(even) {{ background: #f0f0f0; }}
  </style>
</head>
<body>
  <h2>Lab 2 — Step 5: Permutation Importance</h2>
  <p class="subtitle">
    Target: <strong>achievement_score</strong> &nbsp;|&nbsp;
    Baseline MSE: <strong>{baseline_mse:.2f}</strong> &nbsp;|&nbsp;
    Repeats per feature: <strong>{N_REPEATS}</strong>
  </p>
  <p>Each bar shows how much the model's error <em>increased</em> when that
     feature's values were randomly scrambled. A bigger bar = the model
     relied on that feature more.</p>
  <div class="chart">{bars_html}
  </div>
  <table>
    <tr><th>Rank</th><th>Feature</th><th>MSE Increase</th><th>Interpretation</th></tr>
    {"".join(
        f"<tr><td>{i+1}</td><td>{f}</td><td>{v:.3f}</td>"
        f"<td>{'Most important' if i==0 else 'Least important' if i==len(features)-1 else ''}</td></tr>"
        for i,(f,v) in enumerate(zip(features,values))
    )}
  </table>
  <p class="footer">Generated by Lab 2 Step 5 &mdash; Modeling Equity Series</p>
</body>
</html>"""

html_path = os.path.join(base, "permutation_importance.html")
with open(html_path, "w") as f:
    f.write(html)

print(f"\n  HTML chart saved to: {html_path}")
