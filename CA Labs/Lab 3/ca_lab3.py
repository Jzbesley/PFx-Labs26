"""
California Lab 3 — Full Pipeline
Bias check, schema redesign, equity checker before/after,
bonus LR rerun on redesigned data, and CA capstone summary.
"""

import csv, math, os, random, sys
from collections import defaultdict
from datetime import date
random.seed(2024)

base      = os.path.dirname(os.path.abspath(__file__))
lab2_csv  = os.path.join(base, "..", "Lab 2", "ca_lab2_students.csv")
lab1_metrics = os.path.join(base, "..", "Lab 1", "ca_lab1_metrics.csv")
lab2_metrics = os.path.join(base, "..", "Lab 2", "ca_lab2_metrics.csv")
clustered    = os.path.join(base, "..", "Lab 2", "ca_lab2_clustered.csv")

# ── Equity checker (inline, mirrors GA Lab 3) ─────────────────────────────────
SENS_KEYS=["meal","income","economic","need","program","risk","region","district",
           "background","status","ability","support","resource","poverty","socio","free","reduced"]

def check_equity(filepath_or_rows, label=""):
    if isinstance(filepath_or_rows,str):
        rows=list(csv.DictReader(open(filepath_or_rows,newline=""))); src=os.path.basename(filepath_or_rows)
    else:
        rows=filepath_or_rows; src=label
    if not rows: return 0
    cols=list(rows[0].keys()); warnings=[]
    for col in cols:
        vals=[r[col] for r in rows if r[col]!=""]
        unique=sorted(set(vals)); n=len(unique)
        try: [float(v) for v in vals]; is_num=True
        except: is_num=False
        sens=any(k in col.lower() for k in SENS_KEYS)
        if n<=2 and sens: warnings.append(("BINARY",col,f"'{col}' is BINARY {unique[:3]}"))
        elif 2<n<4 and sens and not is_num: warnings.append(("FEW_CAT",col,f"'{col}' has {n} categories"))
    print(f"  check_schema_equity — {src}: {len(warnings)} warning(s)")
    for kind,col,msg in warnings: print(f"    [{kind}] {msg}")
    return len(warnings)

# ── CA Sub-regions ────────────────────────────────────────────────────────────
CA_SUB_REGIONS={
    "Los Angeles USD":     ["LA Central","LA East","LA West","LA South"],
    "San Diego USD":       ["SD North","SD South","SD East"],
    "San Francisco USD":   ["SF Mission","SF Richmond","SF Sunset"],
    "Fresno USD":          ["Fresno Central","Fresno West","Fresno East"],
    "Sacramento City USD": ["Sac Downtown","Sac Suburbs","Sac Valley"],
    "Oakland USD":         ["Oakland Hills","Oakland Flatlands","Oakland North"],
    "San Jose USD":        ["SJ North","SJ South","SJ East"],
    "Bakersfield City SD": ["Bako Central","Bako East","Bako West"],
}

# ── Step 1: Bias Check ────────────────────────────────────────────────────────
print("="*68)
print("  CA Lab 3 — Step 1: Bias Check")
print("="*68)
rows_orig=list(csv.DictReader(open(lab2_csv,newline="")))
cols=list(rows_orig[0].keys())
for col in cols:
    vals=[r[col] for r in rows_orig if r[col]!=""]
    unique=set(vals); n=len(unique)
    try: [float(v) for v in vals]; kind="NUMERIC"
    except: kind="STRING"
    flag="⚠ BINARY SENSITIVE" if n<=2 and any(k in col.lower() for k in SENS_KEYS) else ""
    print(f"  {col:<24} {kind:<8} unique={n:<4} {flag}")
print()

# ── Step 2: SDMX note ────────────────────────────────────────────────────────
sdmx_path=os.path.join(base,"ca_sdmx_reflection.txt")
with open(sdmx_path,"w") as f:
    f.write("CA Lab 3 — SDMX Discussion Notes\n"+"="*60+"\n\n")
    f.write("California uses the Local Control Funding Formula (LCFF)\n")
    f.write("to categorize student need. This is more granular than\n")
    f.write("a simple yes/no meals program flag — it recognizes English\n")
    f.write("learners, foster youth, and low-income students separately.\n\n")
    f.write("Discussion: How does LCFF compare to SDMX standards?\n")
    f.write("What categories does LCFF still oversimplify?\n\n")
    f.write("[Group notes here]\n")
print("  SDMX reflection saved to ca_sdmx_reflection.txt")
print()

# ── Step 3: Redesign Schema ───────────────────────────────────────────────────
def economic_level(row):
    meals=row["meals_program"]=="yes"; ach=float(row["achievement_score"])
    if not meals: return "low" if ach>=65 else "moderate"
    return "critical" if ach<52 else "high"

rows_redesigned=[]
for row in rows_orig:
    new_row=dict(row)
    new_row["economic_need_level"]=economic_level(row)
    new_row["sub_region"]=random.choice(CA_SUB_REGIONS.get(row["district"],["Unknown"]))
    rows_redesigned.append(new_row)

redesign_path=os.path.join(base,"ca_lab3_redesigned.csv")
old_fields=list(rows_orig[0].keys())
new_fields=[]
for f in old_fields:
    if f=="meals_program": new_fields.append("economic_need_level")
    elif f=="district":    new_fields.extend(["district","sub_region"])
    else:                  new_fields.append(f)

with open(redesign_path,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=new_fields,extrasaction="ignore")
    w.writeheader(); w.writerows(rows_redesigned)

level_counts=defaultdict(int)
for r in rows_redesigned: level_counts[r["economic_need_level"]]+=1
print("  Schema Redesign — economic_need_level distribution:")
for lv in ["low","moderate","high","critical"]:
    n=level_counts[lv]; bar="█"*int(n/len(rows_redesigned)*40)
    print(f"    {lv:<10} {n:>4} ({n/len(rows_redesigned)*100:.1f}%)  {bar}")
print()

# ── Step 4 & 5: Before / After equity check ────────────────────────────────────
print("="*68)
print("  CA Lab 3 — Step 5: Before vs. After")
print("="*68)
n_before=check_equity(lab2_csv)
n_after =check_equity(redesign_path)
print(f"\n  Warnings BEFORE: {n_before}  |  AFTER: {n_after}  |  Removed: {n_before-n_after}")
print()

# ── Step 6: Bonus LR rerun on redesigned data ─────────────────────────────────
LEVELS=["low","moderate","high","critical"]
def sigmoid(z): return 1/(1+math.exp(-max(-500,min(500,z))))
def dot(w,x):   return sum(wi*xi for wi,xi in zip(w,x))

rr=[]
with open(redesign_path,newline="") as f:
    for r in csv.DictReader(f):
        att=r["attendance_rate"]; quiz=r["avg_quiz_score"]
        if att=="" or quiz=="": continue
        lv=r["economic_need_level"]; oh=[1.0 if lv==l else 0.0 for l in LEVELS]
        rr.append({"attendance_rate":float(att),"avg_quiz_score":float(quiz),
                   "l0":oh[0],"l1":oh[1],"l2":oh[2],"l3":oh[3],
                   "at_risk":1 if r["at_risk"]=="yes" else 0})

def cs(d,k):
    v=[r[k] for r in d]; mu=sum(v)/len(v)
    sd=math.sqrt(sum((x-mu)**2 for x in v)/len(v)) or 1e-8; return mu,sd
st={k:cs(rr,k) for k in ["attendance_rate","avg_quiz_score"]}
def fv(r): return [(r["attendance_rate"]-st["attendance_rate"][0])/st["attendance_rate"][1],
                   (r["avg_quiz_score"]-st["avg_quiz_score"][0])/st["avg_quiz_score"][1],
                   r["l0"],r["l1"],r["l2"],r["l3"],1.0]
random.shuffle(rr); sp=int(0.8*len(rr))
Xtr=[fv(r) for r in rr[:sp]]; ytr=[r["at_risk"] for r in rr[:sp]]
Xte=[fv(r) for r in rr[sp:]]; yte=[r["at_risk"] for r in rr[sp:]]
w=[0.0]*7
for _ in range(1000):
    g=[0.0]*7
    for x,y in zip(Xtr,ytr):
        e=sigmoid(dot(w,x))-y
        for j in range(7): g[j]+=e*x[j]
    for j in range(7): w[j]-=0.1*g[j]/len(Xtr)
yp=[1 if sigmoid(dot(w,x))>=0.5 else 0 for x in Xte]
TP=sum(1 for p,a in zip(yp,yte) if p==1 and a==1)
FP=sum(1 for p,a in zip(yp,yte) if p==1 and a==0)
FN=sum(1 for p,a in zip(yp,yte) if p==0 and a==1)
TN=sum(1 for p,a in zip(yp,yte) if p==0 and a==0)
n=len(yte)
rd_acc=(TP+TN)/n; rd_pre=TP/(TP+FP) if TP+FP>0 else 0; rd_rec=TP/(TP+FN) if TP+FN>0 else 0

# Load original LR metrics
lr_metrics={}
with open(lab1_metrics,newline="") as f:
    for row in csv.DictReader(f):
        if row["model"]=="logistic_regression":
            lr_metrics=row; break

print("="*66)
print("  CA Lab 3 — Step 6: LR Original vs. Redesigned")
print("="*66)
print(f"  {'Metric':<14} {'Original':>14} {'Redesigned':>14}")
print("  "+"-"*44)
for metric,orig_key in [("Accuracy","accuracy"),("Precision","precision"),("Recall","recall")]:
    orig=float(lr_metrics.get(orig_key,0))
    new=rd_acc if metric=="Accuracy" else (rd_pre if metric=="Precision" else rd_rec)
    print(f"  {metric:<14} {orig:>14.1%} {new:>14.1%}")
print()

# ── Step 7: CA Capstone Summary ───────────────────────────────────────────────
lab2m={}
with open(lab2_metrics,newline="") as f:
    for row in csv.DictReader(f): lab2m[row["key"]]=row["value"]

cl_rows=list(csv.DictReader(open(clustered,newline="")))
cl_data=defaultdict(list)
for r in cl_rows: cl_data[r["cluster"]].append(r)
cl_table=[]
for k,mb in sorted(cl_data.items()):
    avg=sum(float(r["achievement_score"]) for r in mb)/len(mb)
    risk=sum(1 for r in mb if r.get("at_risk","no")=="yes")/len(mb)*100
    meal=sum(1 for r in mb if r.get("meals_program","no")=="yes")/len(mb)*100
    cl_table.append((k,len(mb),round(avg,1),round(risk,1),round(meal,1)))
cl_table.sort(key=lambda x:-x[2])

dist_data2=defaultdict(list)
for r in rows_orig: dist_data2[r["district"]].append(float(r["achievement_score"]))
dist_avgs=sorted([(d,round(sum(v)/len(v),1)) for d,v in dist_data2.items()],key=lambda x:x[1])
low_d=dist_avgs[0]; high_d=dist_avgs[-1]

cluster_md="| Cluster | Students | Avg Score | At-Risk % | Meals % |\n|---|---|---|---|---|\n"
for k,n,ach,risk,meal in cl_table:
    cluster_md+=f"| {k} | {n} | {ach} | {risk}% | {meal}% |\n"

dist_md="| District | Avg Score |\n|---|---|\n"
for d,s in sorted(dist_avgs,key=lambda x:-x[1]):
    dist_md+=f"| {d} | {s} |\n"

lr_acc_f=float(lr_metrics.get("accuracy",0))
lr_pre_f=float(lr_metrics.get("precision",0))
lr_rec_f=float(lr_metrics.get("recall",0))
rf_row={}
with open(lab1_metrics,newline="") as f:
    for row in csv.DictReader(f):
        if row["model"]=="random_forest": rf_row=row; break
rf_acc_f=float(rf_row.get("accuracy",0))
rf_pre_f=float(rf_row.get("precision",0))
rf_rec_f=float(rf_row.get("recall",0))

summary_md=f"""# California Modeling Equity — Full Series Capstone Summary
**Generated:** {date.today().strftime("%B %d, %Y")}  
**State:** California (synthetic data — no real student records used)

---

## Lab 1 — Prediction Tool

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| **Accuracy** | {lr_acc_f:.1%} | {rf_acc_f:.1%} |
| **Precision** | {lr_pre_f:.1%} | {rf_pre_f:.1%} |
| **Recall** | {lr_rec_f:.1%} | {rf_rec_f:.1%} |

---

## Lab 2 — Where the Gaps Show Up

### K-Means Clusters (K=4)

{cluster_md}

### District Achievement Scores

{dist_md}

**Lowest district:** {low_d[0]} (avg: {low_d[1]})  
**Highest district:** {high_d[0]} (avg: {high_d[1]})  
**Gap:** {high_d[1]-low_d[1]:.1f} points

**Top feature (mutual information):** `{lab2m.get('top_feature','?')}` (MI={lab2m.get('top_mi','?')})  
**Top permutation importance feature:** `{lab2m.get('top_perm_feat','?')}` (+{lab2m.get('top_perm_imp','?')} MSE)

---

## Lab 3 — Fixing the Data

| | Warnings |
|---|---|
| Original data | {n_before} |
| Redesigned data | {n_after} |
| Improvement | {n_before-n_after} warning(s) removed |

### Model Performance: Original vs. Redesigned

| Metric | Original | Redesigned |
|---|---|---|
| **Accuracy** | {lr_acc_f:.1%} | {rd_acc:.1%} |
| **Precision** | {lr_pre_f:.1%} | {rd_pre:.1%} |
| **Recall** | {lr_rec_f:.1%} | {rd_rec:.1%} |

---

## Capstone Reflection

**One unfairness in Lab 1:** Using `meals_program` as a binary predictor
means the model treats income as a proxy for academic risk, potentially
over-flagging low-income students in districts like {low_d[0]}.

**One gap found in Lab 2:** A {high_d[1]-low_d[1]:.1f}-point achievement gap exists between
{low_d[0]} and {high_d[0]}, mirroring California's well-documented
resource disparities between wealthy and under-resourced districts.

**One problem fixed in Lab 3:** `meals_program` (binary) was replaced with
`economic_need_level` (4 levels), reducing equity warnings from {n_before} to {n_after}.

**Advice for a school:** Treat the model as one signal — not a decision.
Audit regularly for disparate impact, especially across district lines.

---
*All data is synthetic — no real student records were used.*
"""

summary_path=os.path.join(base,"ca_lab3_summary.md")
with open(summary_path,"w") as f: f.write(summary_md)

# Save CA lab3 metrics for comparison
ca_lab3_metrics=os.path.join(base,"ca_lab3_metrics.csv")
with open(ca_lab3_metrics,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["key","value"])
    w.writerow(["warnings_before",n_before]); w.writerow(["warnings_after",n_after])
    w.writerow(["rd_acc",round(rd_acc,4)]); w.writerow(["rd_pre",round(rd_pre,4)])
    w.writerow(["rd_rec",round(rd_rec,4)]); w.writerow(["district_gap",round(high_d[1]-low_d[1],1)])
    w.writerow(["low_district",low_d[0]]); w.writerow(["low_score",low_d[1]])
    w.writerow(["high_district",high_d[0]]); w.writerow(["high_score",high_d[1]])

print(f"  CA Capstone saved: {summary_path}")
print(f"  District gap: {high_d[1]-low_d[1]:.1f} pts ({low_d[0]} vs {high_d[0]})")
print(f"  Equity warnings: {n_before} → {n_after}")
print("="*68)
