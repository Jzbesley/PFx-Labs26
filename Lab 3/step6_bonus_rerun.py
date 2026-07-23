"""
Lab 3 — Step 6 (Bonus): Re-run Logistic Regression on Redesigned Data
Replaces the binary meals_program with one-hot encoded economic_need_level
and compares metrics to the original Lab 1 results.
"""

import csv, math, os, random
random.seed(42)

base        = os.path.dirname(os.path.abspath(__file__))
redesign_csv = os.path.join(base, "lab3_students_redesigned.csv")

# ── Load redesigned data ──────────────────────────────────────────────────────
LEVELS = ["low", "moderate", "high", "critical"]

rows = []
with open(redesign_csv, newline="") as f:
    for r in csv.DictReader(f):
        att  = r["attendance_rate"]
        quiz = r["avg_quiz_score"]
        if att == "" or quiz == "":
            continue
        # One-hot encode economic_need_level
        level = r["economic_need_level"]
        onehot = [1.0 if level == lv else 0.0 for lv in LEVELS]
        rows.append({
            "attendance_rate": float(att),
            "avg_quiz_score":  float(quiz),
            "level_low":      onehot[0],
            "level_moderate": onehot[1],
            "level_high":     onehot[2],
            "level_critical": onehot[3],
            "at_risk": 1 if r["at_risk"] == "yes" else 0,
        })

FEAT_KEYS = ["attendance_rate", "avg_quiz_score",
             "level_low", "level_moderate", "level_high", "level_critical"]

# ── Standardize numeric features (leave one-hot as-is) ───────────────────────
def col_stats(data, key):
    vals = [d[key] for d in data]
    mu   = sum(vals) / len(vals)
    sd   = math.sqrt(sum((v-mu)**2 for v in vals) / len(vals)) or 1e-8
    return mu, sd

numeric_keys = ["attendance_rate", "avg_quiz_score"]
stats = {k: col_stats(rows, k) for k in numeric_keys}

def feats(row):
    vec = []
    for k in FEAT_KEYS:
        if k in stats:
            vec.append((row[k] - stats[k][0]) / stats[k][1])
        else:
            vec.append(row[k])
    vec.append(1.0)   # bias
    return vec

# ── 80/20 split ───────────────────────────────────────────────────────────────
random.shuffle(rows)
split      = int(0.8 * len(rows))
train_rows = rows[:split]
test_rows  = rows[split:]

X_train = [feats(r) for r in train_rows]
y_train = [r["at_risk"] for r in train_rows]
X_test  = [feats(r) for r in test_rows]
y_test  = [r["at_risk"] for r in test_rows]

# ── Logistic Regression ───────────────────────────────────────────────────────
def sigmoid(z): return 1/(1+math.exp(-max(-500,min(500,z))))
def dot(w,x):   return sum(wi*xi for wi,xi in zip(w,x))

n_feat  = len(X_train[0])
weights = [0.0] * n_feat

for _ in range(1000):
    grad = [0.0] * n_feat
    for x, y in zip(X_train, y_train):
        err = sigmoid(dot(weights, x)) - y
        for j in range(n_feat):
            grad[j] += err * x[j]
    for j in range(n_feat):
        weights[j] -= 0.1 * grad[j] / len(X_train)

y_pred = [1 if sigmoid(dot(weights,x)) >= 0.5 else 0 for x in X_test]

TP = sum(1 for p,a in zip(y_pred,y_test) if p==1 and a==1)
FP = sum(1 for p,a in zip(y_pred,y_test) if p==1 and a==0)
FN = sum(1 for p,a in zip(y_pred,y_test) if p==0 and a==1)
TN = sum(1 for p,a in zip(y_pred,y_test) if p==0 and a==0)
n  = len(y_test)

acc  = (TP+TN)/n
pre  = TP/(TP+FP) if (TP+FP)>0 else 0.0
rec  = TP/(TP+FN) if (TP+FN)>0 else 0.0

# Original Lab 1 results
LAB1_ACC, LAB1_PRE, LAB1_REC = 0.881, 0.818, 0.643

print("=" * 66)
print("  Lab 3 — Step 6: Logistic Regression — Original vs. Redesigned")
print("=" * 66)
print(f"  Feature change: meals_program (binary) → economic_need_level")
print(f"  Encoding: one-hot across 4 levels (low/moderate/high/critical)")
print()
print(f"  {'Metric':<14} {'Original (Lab 1)':>18} {'Redesigned (Lab 3)':>20}")
print("  " + "-"*54)
print(f"  {'Accuracy':<14} {LAB1_ACC:>18.1%} {acc:>20.1%}")
print(f"  {'Precision':<14} {LAB1_PRE:>18.1%} {pre:>20.1%}")
print(f"  {'Recall':<14} {LAB1_REC:>18.1%} {rec:>20.1%}")
print("=" * 66)
print()
print("  Confusion Matrix (redesigned data):")
print(f"  {'':28} Predicted NO   Predicted YES")
print(f"  {'Actually NOT at-risk':<28} TN={TN:<6}        FP={FP}")
print(f"  {'Actually AT RISK':<28} FN={FN:<6}        TP={TP}")
print("=" * 66)
print()
diff_acc = acc - LAB1_ACC
diff_pre = pre - LAB1_PRE
diff_rec = rec - LAB1_REC
print("  Interpretation:")
print(f"  Accuracy  changed by {diff_acc:+.1%}")
print(f"  Precision changed by {diff_pre:+.1%}")
print(f"  Recall    changed by {diff_rec:+.1%}")
print()
print("  Using 4 economic need levels instead of a binary yes/no gives")
print("  the model more nuanced information about each student's context,")
print("  which can improve fairness without sacrificing performance.")
