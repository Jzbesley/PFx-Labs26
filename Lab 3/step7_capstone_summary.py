"""
Lab 3 — Step 7: Georgia Capstone Summary
Generates lab3_georgia_summary.md pulling results from all 3 labs.
"""

import csv, os, math, random
from datetime import date
random.seed(42)

base = os.path.dirname(os.path.abspath(__file__))

# ── Pull Lab 1 metrics (recompute live) ───────────────────────────────────────
def sigmoid(z): return 1/(1+math.exp(-max(-500,min(500,z))))
def dot(w,x):   return sum(wi*xi for wi,xi in zip(w,x))

def run_logistic(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            att = r["attendance_rate"]; quiz = r["avg_quiz_score"]
            if att == "" or quiz == "": continue
            rows.append({"attendance_rate":float(att),"avg_quiz_score":float(quiz),
                         "meals_program":1 if r["meals_program"]=="yes" else 0,
                         "at_risk":1 if r["at_risk"]=="yes" else 0})
    def cs(d,k):
        v=[r[k] for r in d]; mu=sum(v)/len(v)
        sd=math.sqrt(sum((x-mu)**2 for x in v)/len(v)) or 1e-8; return mu,sd
    stats={k:cs(rows,k) for k in ["attendance_rate","avg_quiz_score"]}
    def feats(r):
        return [(r["attendance_rate"]-stats["attendance_rate"][0])/stats["attendance_rate"][1],
                (r["avg_quiz_score"]-stats["avg_quiz_score"][0])/stats["avg_quiz_score"][1],
                r["meals_program"],1.0]
    random.shuffle(rows); split=int(0.8*len(rows))
    Xtr=[feats(r) for r in rows[:split]]; ytr=[r["at_risk"] for r in rows[:split]]
    Xte=[feats(r) for r in rows[split:]]; yte=[r["at_risk"] for r in rows[split:]]
    w=[0.0]*4
    for _ in range(1000):
        g=[0.0]*4
        for x,y in zip(Xtr,ytr):
            e=sigmoid(dot(w,x))-y
            for j in range(4): g[j]+=e*x[j]
        for j in range(4): w[j]-=0.1*g[j]/len(Xtr)
    yp=[1 if sigmoid(dot(w,x))>=0.5 else 0 for x in Xte]
    TP=sum(1 for p,a in zip(yp,yte) if p==1 and a==1)
    FP=sum(1 for p,a in zip(yp,yte) if p==1 and a==0)
    FN=sum(1 for p,a in zip(yp,yte) if p==0 and a==1)
    TN=sum(1 for p,a in zip(yp,yte) if p==0 and a==0)
    n=len(yte)
    return ((TP+TN)/n, TP/(TP+FP) if TP+FP>0 else 0, TP/(TP+FN) if TP+FN>0 else 0,
            sum(1 for r in rows if r["at_risk"]==1), len(rows))

lab1_clean = os.path.join(base,"..","Lab 1","students_clean.csv")
lr_acc,lr_pre,lr_rec,n_atrisk,n_total = run_logistic(lab1_clean)
RF_ACC,RF_PRE,RF_REC = 0.814, 1.000, 0.214

# ── Pull Lab 2 cluster data ────────────────────────────────────────────────────
from collections import defaultdict
cluster_rows = list(csv.DictReader(open(
    os.path.join(base,"..","Lab 2","lab2_students_clustered.csv"))))
cluster_data = defaultdict(list)
for r in cluster_rows: cluster_data[r["cluster"]].append(r)
cluster_table = []
for k,members in sorted(cluster_data.items()):
    avg_ach  = sum(float(r["achievement_score"]) for r in members)/len(members)
    pct_risk = sum(1 for r in members if r["at_risk"]=="yes")/len(members)*100
    pct_meal = sum(1 for r in members if r["meals_program"]=="yes")/len(members)*100
    cluster_table.append((k, len(members), round(avg_ach,1), round(pct_risk,1), round(pct_meal,1)))
cluster_table.sort(key=lambda x: -x[2])

dist_rows = list(csv.DictReader(open(
    os.path.join(base,"..","Lab 2","lab2_students.csv"))))
dist_avgs = defaultdict(list)
for r in dist_rows: dist_avgs[r["district"]].append(float(r["achievement_score"]))
dist_summary = sorted(
    [(d, round(sum(v)/len(v),1)) for d,v in dist_avgs.items()],
    key=lambda x: x[1])
lowest_dist  = dist_summary[0]
highest_dist = dist_summary[-1]

# ── Pull Lab 3 warning counts ─────────────────────────────────────────────────
# Re-run checker inline to get counts without printing
def count_warnings(filepath):
    rows = list(csv.DictReader(open(filepath, newline="")))
    if not rows: return 0
    SENS = ["meal","income","economic","need","program","risk","region",
            "district","background","status","ability","support","resource",
            "poverty","socio","free","reduced"]
    w = 0
    for col in rows[0].keys():
        vals=[r[col] for r in rows if r[col]!=""]
        unique=set(vals); n=len(unique)
        try: [float(v) for v in vals]; is_num=True
        except: is_num=False
        sens=any(k in col.lower() for k in SENS)
        if n<=2 and sens: w+=1
        elif 2<n<4 and sens and not is_num: w+=1
    return w

n_warn_before = count_warnings(os.path.join(base,"..","Lab 2","lab2_students.csv"))
n_warn_after  = count_warnings(os.path.join(base,"lab3_students_redesigned.csv"))

# ── Redesigned LR metrics ─────────────────────────────────────────────────────
redesign_csv = os.path.join(base,"lab3_students_redesigned.csv")
LEVELS=["low","moderate","high","critical"]
rows_rd=[]
with open(redesign_csv,newline="") as f:
    for r in csv.DictReader(f):
        att=r["attendance_rate"]; quiz=r["avg_quiz_score"]
        if att=="" or quiz=="": continue
        lv=r["economic_need_level"]; oh=[1.0 if lv==l else 0.0 for l in LEVELS]
        rows_rd.append({"attendance_rate":float(att),"avg_quiz_score":float(quiz),
                        "l0":oh[0],"l1":oh[1],"l2":oh[2],"l3":oh[3],
                        "at_risk":1 if r["at_risk"]=="yes" else 0})
def cs2(d,k):
    v=[r[k] for r in d]; mu=sum(v)/len(v)
    sd=math.sqrt(sum((x-mu)**2 for x in v)/len(v)) or 1e-8; return mu,sd
st2={k:cs2(rows_rd,k) for k in ["attendance_rate","avg_quiz_score"]}
def f2(r):
    return [(r["attendance_rate"]-st2["attendance_rate"][0])/st2["attendance_rate"][1],
            (r["avg_quiz_score"]-st2["avg_quiz_score"][0])/st2["avg_quiz_score"][1],
            r["l0"],r["l1"],r["l2"],r["l3"],1.0]
random.shuffle(rows_rd); sp=int(0.8*len(rows_rd))
Xtr2=[f2(r) for r in rows_rd[:sp]]; ytr2=[r["at_risk"] for r in rows_rd[:sp]]
Xte2=[f2(r) for r in rows_rd[sp:]]; yte2=[r["at_risk"] for r in rows_rd[sp:]]
w2=[0.0]*7
for _ in range(1000):
    g=[0.0]*7
    for x,y in zip(Xtr2,ytr2):
        e=sigmoid(dot(w2,x))-y
        for j in range(7): g[j]+=e*x[j]
    for j in range(7): w2[j]-=0.1*g[j]/len(Xtr2)
yp2=[1 if sigmoid(dot(w2,x))>=0.5 else 0 for x in Xte2]
TP2=sum(1 for p,a in zip(yp2,yte2) if p==1 and a==1)
FP2=sum(1 for p,a in zip(yp2,yte2) if p==1 and a==0)
FN2=sum(1 for p,a in zip(yp2,yte2) if p==0 and a==1)
TN2=sum(1 for p,a in zip(yp2,yte2) if p==0 and a==0)
n2=len(yte2)
rd_acc=(TP2+TN2)/n2; rd_pre=TP2/(TP2+FP2) if TP2+FP2>0 else 0; rd_rec=TP2/(TP2+FN2) if TP2+FN2>0 else 0

# ── Write summary markdown ────────────────────────────────────────────────────
out_path = os.path.join(base,"lab3_georgia_summary.md")

cluster_md = "| Cluster | Students | Avg Score | At-Risk % | Meals % |\n|---|---|---|---|---|\n"
for k,n,ach,risk,meal in cluster_table:
    cluster_md += f"| {k} | {n} | {ach} | {risk}% | {meal}% |\n"

dist_md = "| District | Avg Score |\n|---|---|\n"
for d,s in sorted(dist_summary, key=lambda x:-x[1]):
    dist_md += f"| {d} | {s} |\n"

md = f"""# Georgia Modeling Equity — Full Series Capstone Summary
**Generated:** {date.today().strftime("%B %d, %Y")}  
**State:** Georgia (synthetic data — no real student records used)

---

## Lab 1 — Prediction Tool

Built a Logistic Regression and Random Forest to predict student at-risk status
using attendance, quiz scores, and meals program participation.

- Dataset: **{n_total} students**, {n_atrisk} at-risk ({n_atrisk/n_total*100:.1f}%)
- Features: `attendance_rate`, `avg_quiz_score`, `meals_program`

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| **Accuracy** | {lr_acc:.1%} | {RF_ACC:.1%} |
| **Precision** | {lr_pre:.1%} | {RF_PRE:.1%} |
| **Recall** | {lr_rec:.1%} | {RF_REC:.1%} |

**One way this tool could have been unfair:**  
Using `meals_program` as a feature means the model partially predicts risk based
on a family's income rather than solely on academic behavior. This can cause
low-income students to be flagged as at-risk even when their attendance and
scores don't warrant it — a form of proxy bias.

---

## Lab 2 — Where the Gaps Show Up

Added district context, achievement scores, and ran deeper analysis.

### K-Means Clusters (K=4)

{cluster_md}
The cluster with the lowest achievement score also had the highest at-risk rate
and meals program participation — confirming the achievement gap is not random.

### District Achievement Scores

{dist_md}
**Lowest district:** {lowest_dist[0]} (avg score: {lowest_dist[1]})  
**Highest district:** {highest_dist[0]} (avg score: {highest_dist[1]})  
**Gap:** {highest_dist[1] - lowest_dist[1]:.1f} points

**Permutation importance finding:**  
`meals_program` was the feature the neural network relied on most
(MSE increase of +62.81 when scrambled) — even though it was only 4th
by mutual information. The model learned income as a shortcut for achievement,
which is exactly the kind of embedded bias that causes harm at scale.

---

## Lab 3 — Fixing the Data

### Schema Redesign

| Old Column | Problem | New Column |
|---|---|---|
| `meals_program` (yes/no) | Binary — squashes income spectrum to 2 boxes | `economic_need_level` (low/moderate/high/critical) |
| `district` (single label) | Hides within-district community differences | `district` + `sub_region` |

### Equity Checker Results

| | Warnings |
|---|---|
| Original data (Lab 2) | {n_warn_before} |
| Redesigned data (Lab 3) | {n_warn_after} |
| Improvement | {n_warn_before - n_warn_after} warning(s) removed |

### Model Performance: Original vs. Redesigned Schema

| Metric | Original (binary meals) | Redesigned (4-level need) |
|---|---|---|
| **Accuracy** | {lr_acc:.1%} | {rd_acc:.1%} |
| **Precision** | {lr_pre:.1%} | {rd_pre:.1%} |
| **Recall** | {lr_rec:.1%} | {rd_rec:.1%} |

---

## Capstone Reflection

**1. One way the Lab 1 tool could have been unfair:**  
The tool used `meals_program` (a yes/no income proxy) as a predictor. Students
from low-income families could be systematically flagged as at-risk regardless
of their actual academic performance — labeling them before they have a chance
to demonstrate what they can do.

**2. One gap found in Lab 2:**  
{lowest_dist[0]} had an average achievement score of {lowest_dist[1]} compared
to {highest_dist[1]} for {highest_dist[0]} — a {highest_dist[1]-lowest_dist[1]:.1f}-point gap. The
lowest-scoring K-Means cluster had {cluster_table[-1][3]}% at-risk students and
{cluster_table[-1][4]}% on the meals program, showing the gap follows income lines.

**3. One problem fixed in Lab 3:**  
The binary `meals_program` column was replaced with a 4-level
`economic_need_level` scale. This gives the model — and any human reviewing the
data — a more accurate picture of each student's economic context, reducing the
chance that a student is over-simplified into "yes" or "no."

**4. One piece of advice for a school before using a tool like this:**  
No prediction tool should make final decisions about which students receive
support. Use it as one signal among many, involve teachers and counselors in
every decision, audit the model regularly for disparate impact across student
groups, and always ask: who gets hurt if this tool is wrong?

---

*All data is synthetic — no real student records were used in this series.*
"""

with open(out_path,"w") as f: f.write(md)

print("="*66)
print("  Lab 3 — Georgia Capstone Summary Generated")
print("="*66)
print(f"  Saved to: {out_path}")
print(f"  LR original : acc={lr_acc:.1%} pre={lr_pre:.1%} rec={lr_rec:.1%}")
print(f"  LR redesigned: acc={rd_acc:.1%} pre={rd_pre:.1%} rec={rd_rec:.1%}")
print(f"  Equity warnings: {n_warn_before} → {n_warn_after}")
print(f"  District gap: {highest_dist[1]-lowest_dist[1]:.1f} pts "
      f"({lowest_dist[0]} vs {highest_dist[0]})")
print("="*66)
