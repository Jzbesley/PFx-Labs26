"""
Lab 2 — Step 6: District Achievement Map
Builds a self-contained HTML map showing average achievement_score
for each of the 8 districts, plotted as colour-coded circles over Georgia.

No external libraries needed — generates vanilla HTML + inline JavaScript
using the free Leaflet.js CDN (loaded at runtime from a public URL).
"""

import csv, os
from collections import defaultdict

base = os.path.dirname(os.path.abspath(__file__))

# ── Load clustered data + merge lat/lon from full dataset ────────────────────
# lab2_students_clustered.csv dropped lat/lon during Step 4 loading,
# so we pull coordinates from lab2_students.csv and join on student_id.
coords = {}
with open(os.path.join(base, "lab2_students.csv"), newline="") as f:
    for r in csv.DictReader(f):
        coords[r["student_id"]] = (float(r["lat"]), float(r["lon"]))

rows = []
with open(os.path.join(base, "lab2_students_clustered.csv"), newline="") as f:
    for r in csv.DictReader(f):
        lat, lon = coords.get(r["student_id"], (32.5, -83.5))
        r["lat"] = lat
        r["lon"] = lon
        rows.append(r)

# ── Aggregate by district ─────────────────────────────────────────────────────
dist_data = defaultdict(lambda: {"scores": [], "lat": None, "lon": None,
                                  "at_risk": 0, "total": 0, "meals": 0})
for r in rows:
    d = r["district"]
    dist_data[d]["scores"].append(float(r["achievement_score"]))
    dist_data[d]["lat"]    = float(r["lat"])
    dist_data[d]["lon"]    = float(r["lon"])
    dist_data[d]["total"] += 1
    if r["at_risk"] == "yes":
        dist_data[d]["at_risk"] += 1
    if r["meals_program"] == "yes":
        dist_data[d]["meals"] += 1

districts = []
for name, info in dist_data.items():
    avg = sum(info["scores"]) / len(info["scores"])
    districts.append({
        "name":         name,
        "lat":          info["lat"],
        "lon":          info["lon"],
        "avg_score":    round(avg, 1),
        "n_students":   info["total"],
        "pct_at_risk":  round(info["at_risk"] / info["total"] * 100, 1),
        "pct_meals":    round(info["meals"]    / info["total"] * 100, 1),
    })

districts.sort(key=lambda d: d["avg_score"])   # lowest first (for color mapping)

# ── Colour scale: red (low) → yellow (mid) → green (high) ────────────────────
scores = [d["avg_score"] for d in districts]
s_min, s_max = min(scores), max(scores)

def score_to_color(score):
    t = (score - s_min) / (s_max - s_min) if s_max > s_min else 0.5
    if t < 0.5:
        r = 220
        g = int(220 * (t * 2))
        b = 0
    else:
        r = int(220 * (1 - (t - 0.5) * 2))
        g = 180
        b = 0
    return f"#{r:02x}{g:02x}{b:02x}"

# ── Build Leaflet markers ─────────────────────────────────────────────────────
markers_js = ""
for d in districts:
    color   = score_to_color(d["avg_score"])
    popup   = (f"{d['name']}<br>"
               f"<b>Avg Score: {d['avg_score']}</b><br>"
               f"Students: {d['n_students']}<br>"
               f"At-risk: {d['pct_at_risk']}%<br>"
               f"Meals program: {d['pct_meals']}%")
    radius  = 18000 + int((d["avg_score"] - s_min) / max(s_max - s_min, 1) * 10000)
    markers_js += f"""
    L.circle([{d['lat']}, {d['lon']}], {{
        color: '{color}',
        fillColor: '{color}',
        fillOpacity: 0.75,
        radius: {radius},
        weight: 2
    }}).bindPopup("{popup}").addTo(map);
    L.marker([{d['lat']}, {d['lon']}]).bindPopup("{popup}").addTo(map);
"""

# ── Legend HTML ───────────────────────────────────────────────────────────────
legend_rows = ""
for d in sorted(districts, key=lambda x: -x["avg_score"]):
    color = score_to_color(d["avg_score"])
    bar_w = int((d["avg_score"] - s_min) / max(s_max - s_min, 1) * 120) + 30
    legend_rows += (
        f"<tr>"
        f"<td><span style='background:{color};display:inline-block;"
        f"width:14px;height:14px;border-radius:50%;'></span> {d['name']}</td>"
        f"<td style='text-align:center;'><strong>{d['avg_score']}</strong></td>"
        f"<td>{d['n_students']}</td>"
        f"<td>{d['pct_at_risk']}%</td>"
        f"<td>{d['pct_meals']}%</td>"
        f"</tr>"
    )

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Lab 2 — District Achievement Map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    body    {{ font-family: Arial, sans-serif; margin: 0; background: #f4f6f8; }}
    h2      {{ text-align: center; color: #2c3e50; padding: 16px 0 4px; margin: 0; }}
    .sub    {{ text-align: center; color: #666; font-size: 0.9em; margin-bottom: 12px; }}
    #map    {{ height: 480px; width: 90%; margin: 0 auto; border-radius: 8px;
               box-shadow: 0 2px 8px rgba(0,0,0,0.2); }}
    .panel  {{ max-width: 800px; margin: 20px auto; background: white;
               border-radius: 8px; padding: 20px;
               box-shadow: 0 2px 6px rgba(0,0,0,0.1); }}
    table   {{ border-collapse: collapse; width: 100%; }}
    th, td  {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
    th      {{ background: #2c3e50; color: white; }}
    tr:nth-child(even) {{ background: #f9f9f9; }}
    .scale  {{ display: flex; align-items: center; gap: 8px;
               margin: 10px auto; width: fit-content; }}
    .grad   {{ width: 200px; height: 14px; border-radius: 4px;
               background: linear-gradient(to right, #dc0000, #dcdc00, #00b400); }}
    .footer {{ text-align: center; font-size: 0.78em; color: #aaa; margin-top: 16px; }}
  </style>
</head>
<body>
  <h2>Lab 2 — District Achievement Score Map</h2>
  <p class="sub">Circle size and colour reflect average achievement score.
     <strong>Click a circle or pin</strong> for details.</p>

  <div id="map"></div>

  <div class="panel">
    <h3 style="margin-top:0;">Colour Scale</h3>
    <div class="scale">
      <span style="color:#c0392b;">Low ({s_min})</span>
      <div class="grad"></div>
      <span style="color:#27ae60;">High ({s_max})</span>
    </div>

    <h3>District Summary Table</h3>
    <table>
      <tr>
        <th>District</th>
        <th>Avg Score</th>
        <th>Students</th>
        <th>At-Risk %</th>
        <th>Meals %</th>
      </tr>
      {legend_rows}
    </table>
    <p class="footer">Lab 2 — Modeling Equity Series &mdash; All data is synthetic.</p>
  </div>

  <script>
    var map = L.map('map').setView([32.5, -83.5], 7);
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        attribution: '© <a href="https://openstreetmap.org">OpenStreetMap</a> contributors'
    }}).addTo(map);
    {markers_js}
  </script>
</body>
</html>"""

# Fix Leaflet URL double-brace escaping: {{ → { in the JS block
# (already handled by using {{s}}, {{z}}, etc. in the f-string above)

map_path = os.path.join(base, "district_map.html")
with open(map_path, "w") as f:
    f.write(html)

print("=" * 62)
print("  Lab 2 — Step 6: District Achievement Map")
print("=" * 62)
print(f"  Map saved to: {map_path}")
print()
print(f"  {'District':<22} {'Avg Score':>10} {'At-Risk %':>11} {'Meals %':>9}")
print("-" * 62)
for d in sorted(districts, key=lambda x: -x["avg_score"]):
    print(f"  {d['name']:<22} {d['avg_score']:>10.1f} "
          f"{d['pct_at_risk']:>10.1f}% {d['pct_meals']:>8.1f}%")
print("=" * 62)
print()
print("  Open district_map.html in your browser to see the interactive map.")
print("  (Requires an internet connection to load the Leaflet map tiles.)")
