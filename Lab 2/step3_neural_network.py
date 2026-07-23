"""
Lab 2 — Step 3: Standardize + Train Neural Network (MLPRegressor)
Predicts achievement_score from numeric features using a two-layer
Multi-Layer Perceptron built from scratch with stdlib only.

Architecture:
  Input (4 features) → Hidden layer 1 (16 neurons, ReLU)
                     → Hidden layer 2 (8 neurons, ReLU)
                     → Output (1 neuron, linear)

Reports Mean Squared Error (MSE) and Root MSE on the test set.
Saves the trained weights to mlp_weights.csv for use in Step 5.
"""

import csv, math, os, random
random.seed(42)

# ── Load data ─────────────────────────────────────────────────────────────────
base = os.path.dirname(os.path.abspath(__file__))
rows = []
with open(os.path.join(base, "lab2_students.csv"), newline="") as f:
    for r in csv.DictReader(f):
        att  = r["attendance_rate"]
        quiz = r["avg_quiz_score"]
        ent  = r["attendance_entropy"]
        if att == "" or quiz == "" or ent == "":
            continue
        rows.append({
            "attendance_rate":    float(att),
            "avg_quiz_score":     float(quiz),
            "attendance_entropy": float(ent),
            "meals_program":      1.0 if r["meals_program"] == "yes" else 0.0,
            "achievement_score":  float(r["achievement_score"]),
        })

# ── Standardize numeric features ─────────────────────────────────────────────
FEAT_KEYS = ["attendance_rate", "avg_quiz_score", "attendance_entropy", "meals_program"]

def col_mean_sd(data, key):
    vals = [d[key] for d in data]
    mu   = sum(vals) / len(vals)
    sd   = math.sqrt(sum((v - mu)**2 for v in vals) / len(vals)) or 1e-8
    return mu, sd

stats = {k: col_mean_sd(rows, k) for k in FEAT_KEYS}

def standardize(row):
    return [(row[k] - stats[k][0]) / stats[k][1] for k in FEAT_KEYS]

# Target: also standardize for stable training, then inverse-transform for MSE
tgt_vals = [r["achievement_score"] for r in rows]
tgt_mu   = sum(tgt_vals) / len(tgt_vals)
tgt_sd   = math.sqrt(sum((v - tgt_mu)**2 for v in tgt_vals) / len(tgt_vals)) or 1e-8

def std_target(v):  return (v - tgt_mu) / tgt_sd
def inv_target(v):  return v * tgt_sd + tgt_mu

# ── 80/20 split ───────────────────────────────────────────────────────────────
random.shuffle(rows)
split      = int(0.8 * len(rows))
train_rows = rows[:split]
test_rows  = rows[split:]

X_train = [standardize(r) for r in train_rows]
y_train = [std_target(r["achievement_score"]) for r in train_rows]
X_test  = [standardize(r) for r in test_rows]
y_test  = [r["achievement_score"] for r in test_rows]   # raw, for final MSE

# ── MLP helpers ───────────────────────────────────────────────────────────────
def relu(x):    return max(0.0, x)
def relu_d(x):  return 1.0 if x > 0 else 0.0

def xavier(fan_in, fan_out):
    """Xavier uniform initialization."""
    limit = math.sqrt(6 / (fan_in + fan_out))
    return [[random.uniform(-limit, limit) for _ in range(fan_in)]
            for _ in range(fan_out)]

def zeros(n): return [0.0] * n

# ── Network architecture: 4 → 16 → 8 → 1 ────────────────────────────────────
W1 = xavier(4, 16);  b1 = zeros(16)
W2 = xavier(16, 8);  b2 = zeros(8)
W3 = xavier(8, 1);   b3 = zeros(1)

def forward(x):
    """Returns (a1, z1, a2, z2, out) for use in backprop."""
    z1 = [sum(W1[j][i]*x[i] for i in range(4))  + b1[j] for j in range(16)]
    a1 = [relu(v) for v in z1]
    z2 = [sum(W2[j][i]*a1[i] for i in range(16)) + b2[j] for j in range(8)]
    a2 = [relu(v) for v in z2]
    out = sum(W3[0][i]*a2[i] for i in range(8)) + b3[0]
    return a1, z1, a2, z2, out

CLIP = 1.0   # gradient clipping threshold

def clip(v):
    return max(-CLIP, min(CLIP, v))

def backprop(x, y_true, lr=0.001):
    a1, z1, a2, z2, out = forward(x)
    # Output layer delta
    d_out = clip(out - y_true)    # MSE derivative + clip
    # Layer 3 gradients
    for i in range(8):
        W3[0][i] -= lr * d_out * a2[i]
    b3[0] -= lr * d_out
    # Layer 2 deltas
    d2 = [clip(relu_d(z2[j]) * sum(W3[k][j] * d_out for k in range(1)))
          for j in range(8)]
    for j in range(8):
        for i in range(16):
            W2[j][i] -= lr * d2[j] * a1[i]
        b2[j] -= lr * d2[j]
    # Layer 1 deltas
    d1 = [clip(relu_d(z1[j]) * sum(W2[k][j] * d2[k] for k in range(8)))
          for j in range(16)]
    for j in range(16):
        for i in range(4):
            W1[j][i] -= lr * d1[j] * x[i]
        b1[j] -= lr * d1[j]

# ── Training loop ─────────────────────────────────────────────────────────────
EPOCHS = 500
n_tr   = len(X_train)
print("  Training neural network (500 epochs)...")
for epoch in range(EPOCHS):
    idx = list(range(n_tr))
    random.shuffle(idx)
    for i in idx:
        backprop(X_train[i], y_train[i])

# ── Evaluate on test set ──────────────────────────────────────────────────────
preds_raw = [inv_target(forward(x)[4]) for x in X_test]
mse  = sum((p - a)**2 for p, a in zip(preds_raw, y_test)) / len(y_test)
rmse = math.sqrt(mse)

# ── Print results ─────────────────────────────────────────────────────────────
print()
print("=" * 62)
print("  Lab 2 — Step 3: Neural Network Results")
print("=" * 62)
print(f"  Architecture  : 4 → 16 → 8 → 1  (ReLU hidden, linear out)")
print(f"  Training rows : {len(X_train)}   Test rows: {len(X_test)}")
print(f"  Epochs        : {EPOCHS}   Learning rate: 0.001  Grad clip: 1.0")
print()
print(f"  Mean Squared Error  (MSE)  : {mse:.2f}")
print(f"  Root MSE            (RMSE) : {rmse:.2f} points")
print()
print(f"  What does this mean?")
print(f"  On average, the neural network's predicted achievement_score")
print(f"  is off by about {rmse:.1f} points (out of 100). For example,")
print(f"  if a student's real score is 65, the model might guess")
print(f"  anywhere between {max(0,65-rmse):.0f} and {min(100,65+rmse):.0f}.")
print()
print(f"  Sample predictions (first 8 test students):")
print(f"  {'Actual':>10} {'Predicted':>12} {'Error':>8}")
print(f"  {'-'*32}")
for actual, pred in zip(y_test[:8], preds_raw[:8]):
    print(f"  {actual:>10.1f} {pred:>12.1f} {abs(pred-actual):>8.1f}")
print("=" * 62)

# ── Save model weights (needed for permutation importance in Step 5) ──────────
# Flatten weights into a single CSV: layer, row, col, value
weights_path = os.path.join(base, "mlp_weights.csv")
with open(weights_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["layer", "row", "col", "value"])
    for j in range(16):
        for i in range(4):
            w.writerow(["W1", j, i, W1[j][i]])
        w.writerow(["b1", j, 0, b1[j]])
    for j in range(8):
        for i in range(16):
            w.writerow(["W2", j, i, W2[j][i]])
        w.writerow(["b2", j, 0, b2[j]])
    for i in range(8):
        w.writerow(["W3", 0, i, W3[0][i]])
    w.writerow(["b3", 0, 0, b3[0]])

# Save standardization params for Step 5
params_path = os.path.join(base, "mlp_params.csv")
with open(params_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["key", "mean", "sd"])
    for k in FEAT_KEYS:
        w.writerow([k, stats[k][0], stats[k][1]])
    w.writerow(["target_mean", tgt_mu, tgt_sd])

print(f"\n  Weights saved to: {weights_path}")
print(f"  Params  saved to: {params_path}")
