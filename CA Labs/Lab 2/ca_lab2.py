"""
California Lab 2 — Full Pipeline
Extends CA Lab 1 data with California-specific districts (8 real metro areas),
then runs feature ranking, neural network, K-Means, permutation importance,
and builds an interactive HTML map.
"""

import csv, math, os, random
from collections import defaultdict
random.seed(2024)

base     = os.path.dirname(os.path.abspath(__file__))
lab1_csv = os.path.join(base, "..", "Lab 1", "ca_students_with_entropy.csv")

# ── California districts (8 real metro areas with approx coordinates) ─────────
CA_DISTRICTS = [
    {"name": "Los Angeles USD",      "lat": 34.052, "lon": -118.243, "base_score": 61},
    {"name": "San Diego USD",        "lat": 32.716, "lon": -117.161, "base_score": 67},
    {"name": "San Francisco USD",    "lat": 37.774, "lon": -122.419, "base_score": 70},
    {"name": "Fresno USD",           "lat": 36.737, "lon": -119.787, "base_score": 54},
    {"name": "Sacramento City USD",  "lat": 38.581, "lon": -121.494, "base_score": 62},
    {"name": "Oakland USD",          "lat": 37.804, "lon": -122.271, "base_score": 57},
    {"name": "San Jose USD",         "lat": 37.338, "lon": -121.886, "base_score": 73},
    {"name": "Bakersfield City SD",  "lat": 35.373, "lon": -119.019, "base_score": 52},
]

CA_SUB_REGIONS = {
    "Los Angeles USD":     ["LA Central","LA East","LA West","LA South"],
    "San Diego USD":       ["SD North","SD South","SD East"],
    "San Francisco USD":   ["SF Mission","SF Richmond","SF Sunset"],
    "Fresno USD":          ["Fresno Central","Fresno West","Fresno East"],
    "Sacramento City USD": ["Sac Downtown","Sac Suburbs","Sac Valley"],
    "Oakland USD":         ["Oakland Hills","Oakland Flatlands","Oakland North"],
    "San Jose USD":        ["SJ North","SJ South","SJ East"],
    "Bakersfield City SD": ["Bako Central","Bako East","Bako West"],
}

def normal_sample(mu, sigma, lo=None, hi=None):
    for _ in range(200):
        u1=random.random() or 1e-10; u2=random.random()
        z=math.sqrt(-2*math.log(u1))*math.cos(2*math.pi*u2)
        v=mu+sigma*z
        if (lo is None or v>=lo) and (hi is None or v<=hi): return round(v,1)
    return round(mu,1)

# ── Step 1: Load and extend ───────────────────────────────────────────────────
rows=[]
with open(lab1_csv,newline="") as f:
    for r in csv.DictReader(f):
        d=random.choice(CA_DISTRICTS)
        gap=random.uniform(8,15) if r["meals_program"]=="yes" else 0
        ach=max(0,min(100,normal_sample(d["base_score"]-gap,8)))
        r["district"]          = d["name"]
        r["achievement_score"] = ach
        r["lat"]               = d["lat"]
        r["lon"]               = d["lon"]
        rows.append(r)

lab2_csv=os.path.join(base,"ca_lab2_students.csv")
new_fields=list(rows[0].keys())
with open(lab2_csv,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=new_fields); w.writeheader(); w.writerows(rows)

# ── Step 2: Feature Ranking (Mutual Information) ──────────────────────────────
y=[1 if r["at_risk"]=="yes" else 0 for r in rows]

FEATS=[
    ("attendance_rate",    lambda r: float(r["attendance_rate"])    if r["attendance_rate"]    !="" else 0.5, 10),
    ("avg_quiz_score",     lambda r: float(r["avg_quiz_score"])     if r["avg_quiz_score"]     !="" else 50, 10),
    ("attendance_entropy", lambda r: float(r["attendance_entropy"]) if r["attendance_entropy"] !="" else 0.5, 10),
    ("achievement_score",  lambda r: float(r["achievement_score"])  if r["achievement_score"]  !="" else 50, 10),
    ("meals_program",      lambda r: 1 if r["meals_program"]=="yes" else 0, 2),
]

def discretize(vals,nb):
    lo,hi=min(vals),max(vals); span=(hi-lo) or 1e-8
    return [min(int((v-lo)/span*nb),nb-1) for v in vals]

def mutual_info(xb,y):
    n=len(y); jt=defaultdict(int); cx=defaultdict(int); cy=defaultdict(int)
    for xi,yi in zip(xb,y): jt[(xi,yi)]+=1; cx[xi]+=1; cy[yi]+=1
    mi=0.0
    for (xi,yi),c in jt.items():
        pxy=c/n; px=cx[xi]/n; py=cy[yi]/n
        if pxy>0 and px>0 and py>0: mi+=pxy*math.log(pxy/(px*py))
    return max(0.0,mi)

rankings=[]
for name,ext,nb in FEATS:
    raw=[ext(r) for r in rows]
    rankings.append((name,mutual_info(discretize(raw,nb),y)))
rankings.sort(key=lambda x:-x[1])

# ── Step 3: Neural Network ────────────────────────────────────────────────────
FEAT_KEYS=["attendance_rate","avg_quiz_score","attendance_entropy","meals_program"]

def col_mu_sd(data,key):
    vals=[d[key] for d in data]; mu=sum(vals)/len(vals)
    sd=math.sqrt(sum((v-mu)**2 for v in vals)/len(vals)) or 1e-8; return mu,sd

nn_rows=[]
for r in rows:
    att=r["attendance_rate"]; quiz=r["avg_quiz_score"]
    if att=="" or quiz=="": continue
    nn_rows.append({"attendance_rate":float(att),"avg_quiz_score":float(quiz),
                    "attendance_entropy":float(r["attendance_entropy"]),
                    "meals_program":1.0 if r["meals_program"]=="yes" else 0.0,
                    "achievement_score":float(r["achievement_score"])})

stats={k:col_mu_sd(nn_rows,k) for k in FEAT_KEYS}
tgt_vals=[r["achievement_score"] for r in nn_rows]
tgt_mu=sum(tgt_vals)/len(tgt_vals)
tgt_sd=math.sqrt(sum((v-tgt_mu)**2 for v in tgt_vals)/len(tgt_vals)) or 1e-8

def std_x(r): return [(r[k]-stats[k][0])/stats[k][1] for k in FEAT_KEYS]
def std_y(v): return (v-tgt_mu)/tgt_sd
def inv_y(v): return v*tgt_sd+tgt_mu
def relu(x):  return max(0.0,x)
def relu_d(x):return 1.0 if x>0 else 0.0
def xavier(fi,fo):
    lim=math.sqrt(6/(fi+fo))
    return [[random.uniform(-lim,lim) for _ in range(fi)] for _ in range(fo)]

random.shuffle(nn_rows); sp=int(0.8*len(nn_rows))
Xtr=[std_x(r) for r in nn_rows[:sp]]; ytr=[std_y(r["achievement_score"]) for r in nn_rows[:sp]]
Xte=[std_x(r) for r in nn_rows[sp:]]; yte=[r["achievement_score"] for r in nn_rows[sp:]]

W1=xavier(4,16); b1=[0.0]*16
W2=xavier(16,8); b2=[0.0]*8
W3=xavier(8,1);  b3=[0.0]

def forward(x):
    z1=[sum(W1[j][i]*x[i] for i in range(4))+b1[j] for j in range(16)]
    a1=[relu(v) for v in z1]
    z2=[sum(W2[j][i]*a1[i] for i in range(16))+b2[j] for j in range(8)]
    a2=[relu(v) for v in z2]
    out=sum(W3[0][i]*a2[i] for i in range(8))+b3[0]
    return a1,z1,a2,z2,out

CLIP=1.0
def backprop(x,yt,lr=0.001):
    a1,z1,a2,z2,out=forward(x)
    d_out=max(-CLIP,min(CLIP,out-yt))
    for i in range(8): W3[0][i]-=lr*d_out*a2[i]
    b3[0]-=lr*d_out
    d2=[max(-CLIP,min(CLIP,relu_d(z2[j])*sum(W3[k][j]*d_out for k in range(1)))) for j in range(8)]
    for j in range(8):
        for i in range(16): W2[j][i]-=lr*d2[j]*a1[i]
        b2[j]-=lr*d2[j]
    d1=[max(-CLIP,min(CLIP,relu_d(z1[j])*sum(W2[k][j]*d2[k] for k in range(8)))) for j in range(16)]
    for j in range(16):
        for i in range(4): W1[j][i]-=lr*d1[j]*x[i]
        b1[j]-=lr*d1[j]

for _ in range(500):
    idx=list(range(len(Xtr))); random.shuffle(idx)
    for i in idx: backprop(Xtr[i],ytr[i])

preds_raw=[inv_y(forward(x)[4]) for x in Xte]
mse=sum((p-a)**2 for p,a in zip(preds_raw,yte))/len(yte)
rmse=math.sqrt(mse)

# Save MLP params for permutation importance
params_path=os.path.join(base,"ca_mlp_params.csv")
with open(params_path,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["key","mean","sd"])
    for k in FEAT_KEYS: w.writerow([k,stats[k][0],stats[k][1]])
    w.writerow(["target_mean",tgt_mu,tgt_sd])

weights_path=os.path.join(base,"ca_mlp_weights.csv")
with open(weights_path,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["layer","row","col","value"])
    for j in range(16):
        for i in range(4): w.writerow(["W1",j,i,W1[j][i]])
        w.writerow(["b1",j,0,b1[j]])
    for j in range(8):
        for i in range(16): w.writerow(["W2",j,i,W2[j][i]])
        w.writerow(["b2",j,0,b2[j]])
    for i in range(8): w.writerow(["W3",0,i,W3[0][i]])
    w.writerow(["b3",0,0,b3[0]])

# ── Step 4: K-Means ───────────────────────────────────────────────────────────
km_rows=[r for r in nn_rows]
KM_FEAT_KEYS=["attendance_rate","avg_quiz_score","attendance_entropy","meals_program","achievement_score"]
km_stats={k:col_mu_sd(km_rows,k) for k in KM_FEAT_KEYS}
def std_km(r): return [(r[k]-km_stats[k][0])/km_stats[k][1] for k in KM_FEAT_KEYS]
X_km=[std_km(r) for r in km_rows]; K=4; d_km=5

def dist2(a,b): return sum((ai-bi)**2 for ai,bi in zip(a,b))

centers=[random.choice(X_km)]
for _ in range(K-1):
    ds=[min(dist2(x,c) for c in centers) for x in X_km]
    total=sum(ds); rv=random.uniform(0,total); run=0
    for x,dd in zip(X_km,ds):
        run+=dd
        if run>=rv: centers.append(x); break
    else: centers.append(X_km[-1])
centers=[list(c) for c in centers]

for _ in range(100):
    labels=[min(range(K),key=lambda k:dist2(x,centers[k])) for x in X_km]
    new_c=[]
    for ki in range(K):
        mb=[X_km[i] for i in range(len(X_km)) if labels[i]==ki]
        new_c.append([sum(m[j] for m in mb)/len(mb) for j in range(d_km)] if mb else centers[ki])
    if new_c==centers: break
    centers=new_c

for r,lbl in zip(km_rows,labels): r["cluster"]=lbl

cluster_data=defaultdict(list)
for r in km_rows: cluster_data[r["cluster"]].append(r)

# ── Step 5: Permutation Importance ────────────────────────────────────────────
random.seed(99)
test_nn=nn_rows[sp:]
X_te_std=[std_x(r) for r in test_nn]
y_te_raw=[r["achievement_score"] for r in test_nn]

def mse_fn(preds,actuals): return sum((p-a)**2 for p,a in zip(preds,actuals))/len(actuals)

bl_preds=[inv_y(forward(x)[4]) for x in X_te_std]
bl_mse=mse_fn(bl_preds,y_te_raw)

importances={}
for fi,feat in enumerate(FEAT_KEYS):
    deltas=[]
    for _ in range(10):
        col=[x[fi] for x in X_te_std]; random.shuffle(col)
        Xp=[x[:fi]+[col[i]]+x[fi+1:] for i,x in enumerate(X_te_std)]
        deltas.append(mse_fn([inv_y(forward(x)[4]) for x in Xp],y_te_raw)-bl_mse)
    importances[feat]=sum(deltas)/len(deltas)

sorted_imp=sorted(importances.items(),key=lambda x:-x[1])

# ── Step 6: Build HTML Map ────────────────────────────────────────────────────
dist_data=defaultdict(lambda:{"scores":[],"lat":None,"lon":None,"at_risk":0,"total":0,"meals":0})
for r in rows:
    d=r["district"]
    dist_data[d]["scores"].append(float(r["achievement_score"]))
    dist_data[d]["lat"]=float(r["lat"]); dist_data[d]["lon"]=float(r["lon"])
    dist_data[d]["total"]+=1
    if r["at_risk"]=="yes": dist_data[d]["at_risk"]+=1
    if r["meals_program"]=="yes": dist_data[d]["meals"]+=1

districts_map=[]
for name,info in dist_data.items():
    avg=sum(info["scores"])/len(info["scores"])
    districts_map.append({"name":name,"lat":info["lat"],"lon":info["lon"],
        "avg_score":round(avg,1),"n":info["total"],
        "pct_risk":round(info["at_risk"]/info["total"]*100,1),
        "pct_meals":round(info["meals"]/info["total"]*100,1)})
districts_map.sort(key=lambda d:d["avg_score"])

scores_map=[d["avg_score"] for d in districts_map]
s_min,s_max=min(scores_map),max(scores_map)
def score_color(s):
    t=(s-s_min)/(s_max-s_min) if s_max>s_min else 0.5
    if t<0.5: r=220;g=int(220*(t*2));b=0
    else: r=int(220*(1-(t-0.5)*2));g=180;b=0
    return f"#{r:02x}{g:02x}{b:02x}"

markers_js=""
for d in districts_map:
    color=score_color(d["avg_score"])
    popup=(f"{d['name']}<br><b>Avg Score: {d['avg_score']}</b><br>"
           f"Students: {d['n']}<br>At-risk: {d['pct_risk']}%<br>Meals: {d['pct_meals']}%")
    radius=18000+int((d["avg_score"]-s_min)/max(s_max-s_min,1)*10000)
    markers_js+=(f"\n    L.circle([{d['lat']},{d['lon']}],"
                 f"{{color:'{color}',fillColor:'{color}',fillOpacity:0.75,"
                 f"radius:{radius},weight:2}}).bindPopup(\"{popup}\").addTo(map);\n"
                 f"    L.marker([{d['lat']},{d['lon']}]).bindPopup(\"{popup}\").addTo(map);")

legend_rows=""
for d in sorted(districts_map,key=lambda x:-x["avg_score"]):
    color=score_color(d["avg_score"])
    legend_rows+=(f"<tr><td><span style='background:{color};display:inline-block;"
                  f"width:14px;height:14px;border-radius:50%;'></span> {d['name']}</td>"
                  f"<td style='text-align:center;'><strong>{d['avg_score']}</strong></td>"
                  f"<td>{d['n']}</td><td>{d['pct_risk']}%</td><td>{d['pct_meals']}%</td></tr>")

map_html=f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>CA Lab 2 — District Achievement Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>body{{font-family:Arial,sans-serif;margin:0;background:#f4f6f8;}}
h2{{text-align:center;color:#2c3e50;padding:16px 0 4px;margin:0;}}
.sub{{text-align:center;color:#666;font-size:.9em;margin-bottom:12px;}}
#map{{height:520px;width:90%;margin:0 auto;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.2);}}
.panel{{max-width:820px;margin:20px auto;background:white;border-radius:8px;padding:20px;box-shadow:0 2px 6px rgba(0,0,0,.1);}}
table{{border-collapse:collapse;width:100%;}}th,td{{border:1px solid #ddd;padding:8px 12px;}}
th{{background:#2c3e50;color:white;}}tr:nth-child(even){{background:#f9f9f9;}}
.footer{{text-align:center;font-size:.78em;color:#aaa;margin-top:16px;}}</style></head>
<body><h2>California Lab 2 — District Achievement Score Map</h2>
<p class="sub">Click a circle or pin for details.</p>
<div id="map"></div>
<div class="panel"><h3 style="margin-top:0;">District Summary</h3>
<table><tr><th>District</th><th>Avg Score</th><th>Students</th><th>At-Risk %</th><th>Meals %</th></tr>
{legend_rows}</table>
<p class="footer">CA Lab 2 — Modeling Equity Series — All data is synthetic.</p></div>
<script>var map=L.map('map').setView([37.0,-119.5],6);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
{{attribution:'© <a href="https://openstreetmap.org">OpenStreetMap</a>'}}).addTo(map);
{markers_js}</script></body></html>"""

map_path=os.path.join(base,"ca_district_map.html")
with open(map_path,"w") as f: f.write(map_html)

# Save clustered data
clustered_path=os.path.join(base,"ca_lab2_clustered.csv")
cl_fields=list(km_rows[0].keys())
with open(clustered_path,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=cl_fields); w.writeheader(); w.writerows(km_rows)

# ── Print summary ─────────────────────────────────────────────────────────────
print("="*68)
print("  California Lab 2 — Summary")
print("="*68)
print(f"\n  Feature Rankings (→ at_risk):")
max_mi=rankings[0][1] or 1
for rank,(name,mi) in enumerate(rankings,1):
    bar="█"*int(mi/max_mi*25)
    print(f"    {rank}. {name:<24} {mi:.4f}  {bar}")

print(f"\n  Neural Network: MSE={mse:.2f}  RMSE={rmse:.2f} pts")

print(f"\n  K-Means Clusters (K=4):")
print(f"  {'Cluster':>8} {'Size':>6} {'Avg Ach':>9} {'At-Risk%':>10} {'Meals%':>8}")
for k,members in sorted(cluster_data.items(),
                         key=lambda x:-sum(r["achievement_score"] for r in x[1])/len(x[1])):
    n=len(members)
    avg=sum(r["achievement_score"] for r in members)/n
    risk=sum(1 for r in members if r.get("at_risk","no")=="yes")/n*100
    meal=sum(r["meals_program"] for r in members)/n*100
    print(f"  {k:>8} {n:>6} {avg:>9.1f} {risk:>9.1f}% {meal:>7.1f}%")

print(f"\n  Permutation Importance (→ achievement_score):")
max_imp=sorted_imp[0][1] or 1
for feat,imp in sorted_imp:
    bar="█"*int(imp/max_imp*25)
    print(f"    {feat:<24} {imp:>8.2f}  {bar}")

print(f"\n  District scores: {districts_map[0]['name']}={districts_map[0]['avg_score']} (lowest) "
      f"→ {districts_map[-1]['name']}={districts_map[-1]['avg_score']} (highest)")
print(f"  Map saved: {map_path}")
print("="*68)

# Save metrics for comparison
metrics_path=os.path.join(base,"ca_lab2_metrics.csv")
with open(metrics_path,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["key","value"])
    w.writerow(["mse",round(mse,2)]); w.writerow(["rmse",round(rmse,2)])
    w.writerow(["top_feature",rankings[0][0]]); w.writerow(["top_mi",round(rankings[0][1],4)])
    w.writerow(["district_low",districts_map[0]["name"]])
    w.writerow(["district_low_score",districts_map[0]["avg_score"]])
    w.writerow(["district_high",districts_map[-1]["name"]])
    w.writerow(["district_high_score",districts_map[-1]["avg_score"]])
    w.writerow(["top_perm_feat",sorted_imp[0][0]])
    w.writerow(["top_perm_imp",round(sorted_imp[0][1],2)])
    for k,members in cluster_data.items():
        avg=sum(r["achievement_score"] for r in members)/len(members)
        risk=sum(1 for r in members if r.get("at_risk","no")=="yes")/len(members)*100
        meal=sum(r["meals_program"] for r in members)/len(members)*100
        w.writerow([f"cluster_{k}_avg_ach",round(avg,1)])
        w.writerow([f"cluster_{k}_pct_risk",round(risk,1)])
        w.writerow([f"cluster_{k}_pct_meal",round(meal,1)])
