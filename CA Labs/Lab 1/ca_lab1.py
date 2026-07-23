"""
California Lab 1 — Full Pipeline
Mirrors Georgia Lab 1 but with California-specific demographics:
 - Higher overall attendance (suburban/urban mix)
 - Larger dataset to reflect CA's larger school population
 - LCFF (Local Control Funding Formula) participation instead of meals_program
   but kept as meals_program for cross-state comparability

Steps:
  1. Generate 300-row synthetic CA student dataset
  2. Median imputation on avg_quiz_score
  3. Logistic Regression (at_risk prediction)
  4. Random Forest (compare to LR)
  5. Attendance entropy per student
  6. Save all outputs
"""

import csv, math, os, random
random.seed(2024)

base = os.path.dirname(os.path.abspath(__file__))

# ── Helpers ───────────────────────────────────────────────────────────────────
def clamp(v, lo, hi): return max(lo, min(hi, v))

def normal_sample(mu, sigma, lo=None, hi=None):
    for _ in range(200):
        u1 = random.random() or 1e-10
        u2 = random.random()
        z  = math.sqrt(-2*math.log(u1)) * math.cos(2*math.pi*u2)
        v  = mu + sigma * z
        if (lo is None or v >= lo) and (hi is None or v <= hi):
            return v
    return clamp(mu, lo or mu, hi or mu)

# ── Step 1: Generate Dataset ──────────────────────────────────────────────────
# California profile: ~38% qualify for free/reduced meals (NCES 2023 estimate)
# Slightly higher base attendance than Georgia (more urban resources)
NUM_ROWS              = 300
ATT_MISSING_RATE      = 0.03
QUIZ_MISSING_RATE     = 0.08

rows = []
for i in range(1, NUM_ROWS + 1):
    meals = "yes" if random.random() < 0.38 else "no"
    att   = clamp(round(normal_sample(0.74 if meals=="yes" else 0.87, 0.13), 4), 0.0, 1.0)
    quiz_mu = 42 + att * 53
    quiz  = clamp(round(normal_sample(quiz_mu, 11), 1), 0.0, 100.0)
    score = -4.0 + (1-att)*7.0 + (100-quiz)*0.06
    prob  = 1/(1+math.exp(-score))
    at_risk = "yes" if random.random() < prob else "no"
    rows.append({
        "student_id":      f"CA{i:04d}",
        "attendance_rate": None if random.random() < ATT_MISSING_RATE  else att,
        "avg_quiz_score":  None if random.random() < QUIZ_MISSING_RATE else quiz,
        "meals_program":   meals,
        "at_risk":         at_risk,
    })

# Save raw
raw_path = os.path.join(base, "ca_students.csv")
fields   = ["student_id","attendance_rate","avg_quiz_score","meals_program","at_risk"]
with open(raw_path,"w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

# ── Step 2: Median Imputation ─────────────────────────────────────────────────
quiz_vals = [r["avg_quiz_score"] for r in rows if r["avg_quiz_score"] is not None]
quiz_vals.sort()
median_q  = quiz_vals[len(quiz_vals)//2]

before_missing = sum(1 for r in rows if r["avg_quiz_score"] is None)
for r in rows:
    if r["avg_quiz_score"] is None:
        r["avg_quiz_score"] = median_q

clean_path = os.path.join(base, "ca_students_clean.csv")
with open(clean_path,"w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

# ── Step 3: Logistic Regression ───────────────────────────────────────────────
def sigmoid(z): return 1/(1+math.exp(-max(-500,min(500,z))))
def dot(w,x):   return sum(wi*xi for wi,xi in zip(w,x))

model_rows = [r for r in rows if r["attendance_rate"] is not None]

def col_stats(data, key):
    vals=[r[key] for r in data]; mu=sum(vals)/len(vals)
    sd=math.sqrt(sum((v-mu)**2 for v in vals)/len(vals)) or 1e-8; return mu,sd

att_mu,att_sd   = col_stats(model_rows,"attendance_rate")
quiz_mu,quiz_sd = col_stats(model_rows,"avg_quiz_score")

def feats_lr(r):
    return [(r["attendance_rate"]-att_mu)/att_sd,
            (r["avg_quiz_score"]-quiz_mu)/quiz_sd,
            1.0 if r["meals_program"]=="yes" else 0.0, 1.0]

random.shuffle(model_rows)
split  = int(0.8*len(model_rows))
Xtr = [feats_lr(r) for r in model_rows[:split]]; ytr=[1 if r["at_risk"]=="yes" else 0 for r in model_rows[:split]]
Xte = [feats_lr(r) for r in model_rows[split:]]; yte=[1 if r["at_risk"]=="yes" else 0 for r in model_rows[split:]]

wlr=[0.0]*4
for _ in range(1000):
    g=[0.0]*4
    for x,y in zip(Xtr,ytr):
        e=sigmoid(dot(wlr,x))-y
        for j in range(4): g[j]+=e*x[j]
    for j in range(4): wlr[j]-=0.1*g[j]/len(Xtr)

yp_lr=[1 if sigmoid(dot(wlr,x))>=0.5 else 0 for x in Xte]

def metrics(yp,ya):
    TP=sum(1 for p,a in zip(yp,ya) if p==1 and a==1)
    FP=sum(1 for p,a in zip(yp,ya) if p==1 and a==0)
    FN=sum(1 for p,a in zip(yp,ya) if p==0 and a==1)
    TN=sum(1 for p,a in zip(yp,ya) if p==0 and a==0)
    n=len(ya)
    return (TP+TN)/n, TP/(TP+FP) if TP+FP>0 else 0, TP/(TP+FN) if TP+FN>0 else 0, TP,FP,FN,TN

lr_acc,lr_pre,lr_rec,lr_TP,lr_FP,lr_FN,lr_TN = metrics(yp_lr,yte)

# ── Step 4: Random Forest ─────────────────────────────────────────────────────
def feats_rf(r): return [r["attendance_rate"],r["avg_quiz_score"],
                          1.0 if r["meals_program"]=="yes" else 0.0]

def gini(labels):
    n=len(labels)
    if n==0: return 0.0
    p=sum(labels)/n; return 1-p**2-(1-p)**2

def best_split_rf(X,y,fi_list):
    best_g,best_f,best_t=-1,None,None; pg=gini(y); n=len(y)
    for fi in fi_list:
        vals=sorted(set(x[fi] for x in X))
        for i in range(len(vals)-1):
            th=(vals[i]+vals[i+1])/2
            ly=[y[k] for k in range(n) if X[k][fi]<=th]
            ry=[y[k] for k in range(n) if X[k][fi]>th]
            if not ly or not ry: continue
            g=pg-(len(ly)/n*gini(ly)+len(ry)/n*gini(ry))
            if g>best_g: best_g,best_f,best_t=g,fi,th
    if best_f is None: return None
    lm=[i for i in range(n) if X[i][best_f]<=best_t]
    rm=[i for i in range(n) if X[i][best_f]>best_t]
    return best_f,best_t,lm,rm

def build_tree_rf(X,y,fi,md,ms):
    if len(y)<ms or md==0 or len(set(y))==1: return int(round(sum(y)/len(y)))
    r=best_split_rf(X,y,fi)
    if r is None: return int(round(sum(y)/len(y)))
    f,t,li,ri=r
    return {"f":f,"t":t,
            "l":build_tree_rf([X[i] for i in li],[y[i] for i in li],fi,md-1,ms),
            "r":build_tree_rf([X[i] for i in ri],[y[i] for i in ri],fi,md-1,ms)}

def pred_tree(node,x):
    if isinstance(node,int): return node
    return pred_tree(node["l"],x) if x[node["f"]]<=node["t"] else pred_tree(node["r"],x)

Xtr_rf=[feats_rf(r) for r in model_rows[:split]]
Xte_rf=[feats_rf(r) for r in model_rows[split:]]
trees=[]
for _ in range(100):
    idx=[random.randint(0,split-1) for _ in range(split)]
    Xb=[Xtr_rf[i] for i in idx]; yb=[ytr[i] for i in idx]
    fi=random.sample(range(3),2)
    trees.append(build_tree_rf(Xb,yb,fi,6,5))

yp_rf=[1 if sum(pred_tree(t,x) for t in trees)>len(trees)/2 else 0 for x in Xte_rf]
rf_acc,rf_pre,rf_rec,rf_TP,rf_FP,rf_FN,rf_TN = metrics(yp_rf,yte)

# ── Step 5: Attendance Entropy ────────────────────────────────────────────────
PERIODS=6; BUCKETS=5

def bucket(r):
    if r<0.2: return 0
    elif r<0.4: return 1
    elif r<0.6: return 2
    elif r<0.8: return 3
    return 4

def entropy(counts):
    total=sum(counts)
    if total==0: return 0.0
    h=0.0
    for c in counts:
        if c>0:
            p=c/total; h-=p*math.log(p)
    return h

def sim_periods(base_rate):
    rates=[]
    for _ in range(PERIODS):
        u1=random.random() or 1e-10; u2=random.random()
        z=math.sqrt(-2*math.log(u1))*math.cos(2*math.pi*u2)
        rates.append(max(0.0,min(1.0,base_rate+0.12*z)))
    return rates

entropy_rows=[]
for r in rows:
    base_att=float(r["attendance_rate"]) if r["attendance_rate"] is not None else 0.5
    periods=sim_periods(base_att)
    counts=[0]*BUCKETS
    for p in periods: counts[bucket(p)]+=1
    h=round(entropy(counts),4)
    entropy_rows.append({**{k:r[k] for k in fields}, "attendance_entropy":h})

entropy_path=os.path.join(base,"ca_students_with_entropy.csv")
with open(entropy_path,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields+["attendance_entropy"])
    w.writeheader(); w.writerows(entropy_rows)

# ── Print summary ─────────────────────────────────────────────────────────────
n_at_risk=sum(1 for r in rows if r["at_risk"]=="yes")
n_meals  =sum(1 for r in rows if r["meals_program"]=="yes")
ent_vals =[r["attendance_entropy"] for r in entropy_rows]

print("="*66)
print("  California Lab 1 — Summary")
print("="*66)
print(f"  Students          : {NUM_ROWS}")
print(f"  At-risk           : {n_at_risk} ({n_at_risk/NUM_ROWS*100:.1f}%)")
print(f"  Meals program     : {n_meals} ({n_meals/NUM_ROWS*100:.1f}%)")
print(f"  Quiz missing (raw): {before_missing}  → imputed with median={median_q:.1f}")
print()
print(f"  {'Metric':<12} {'Logistic Reg':>14} {'Random Forest':>15}")
print("  "+"-"*44)
print(f"  {'Accuracy':<12} {lr_acc:>14.1%} {rf_acc:>15.1%}")
print(f"  {'Precision':<12} {lr_pre:>14.1%} {rf_pre:>15.1%}")
print(f"  {'Recall':<12} {lr_rec:>14.1%} {rf_rec:>15.1%}")
print()
print(f"  Entropy — min:{min(ent_vals):.4f}  max:{max(ent_vals):.4f}  "
      f"mean:{sum(ent_vals)/len(ent_vals):.4f}")
print()
print(f"  Outputs:")
print(f"    {raw_path}")
print(f"    {clean_path}")
print(f"    {entropy_path}")
print("="*66)

# Save metrics for comparison report
metrics_path=os.path.join(base,"ca_lab1_metrics.csv")
with open(metrics_path,"w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["model","accuracy","precision","recall","TP","FP","FN","TN"])
    w.writerow(["logistic_regression",round(lr_acc,4),round(lr_pre,4),round(lr_rec,4),lr_TP,lr_FP,lr_FN,lr_TN])
    w.writerow(["random_forest",round(rf_acc,4),round(rf_pre,4),round(rf_rec,4),rf_TP,rf_FP,rf_FN,rf_TN])
print(f"  Metrics saved: {metrics_path}")
