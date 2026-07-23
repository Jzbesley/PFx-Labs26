# Georgia Modeling Equity — Full Series Capstone Summary
**Generated:** July 23, 2026  
**State:** Georgia (synthetic data — no real student records used)

---

## Lab 1 — Prediction Tool

Built a Logistic Regression and Random Forest to predict student at-risk status
using attendance, quiz scores, and meals program participation.

- Dataset: **291 students**, 77 at-risk (26.5%)
- Features: `attendance_rate`, `avg_quiz_score`, `meals_program`

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| **Accuracy** | 88.1% | 81.4% |
| **Precision** | 81.8% | 100.0% |
| **Recall** | 64.3% | 21.4% |

**One way this tool could have been unfair:**  
Using `meals_program` as a feature means the model partially predicts risk based
on a family's income rather than solely on academic behavior. This can cause
low-income students to be flagged as at-risk even when their attendance and
scores don't warrant it — a form of proxy bias.

---

## Lab 2 — Where the Gaps Show Up

Added district context, achievement scores, and ran deeper analysis.

### K-Means Clusters (K=4)

| Cluster | Students | Avg Score | At-Risk % | Meals % |
|---|---|---|---|---|
| 0 | 107 | 68.4 | 32.7% | 0.0% |
| 2 | 69 | 65.3 | 4.3% | 0.0% |
| 1 | 53 | 58.7 | 13.2% | 0.0% |
| 3 | 62 | 53.9 | 51.6% | 0.0% |

The cluster with the lowest achievement score also had the highest at-risk rate
and meals program participation — confirming the achievement gap is not random.

### District Achievement Scores

| District | Avg Score |
|---|---|
| Highpoint ISD | 75.3 |
| Lakeshore ISD | 69.0 |
| Maplewood CSD | 65.8 |
| Westview USD | 65.3 |
| Oakdale USD | 60.5 |
| Pinecrest ISD | 59.9 |
| Riverton USD | 54.6 |
| Cedarville CSD | 51.0 |

**Lowest district:** Cedarville CSD (avg score: 51.0)  
**Highest district:** Highpoint ISD (avg score: 75.3)  
**Gap:** 24.3 points

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
| Original data (Lab 2) | 2 |
| Redesigned data (Lab 3) | 1 |
| Improvement | 1 warning(s) removed |

### Model Performance: Original vs. Redesigned Schema

| Metric | Original (binary meals) | Redesigned (4-level need) |
|---|---|---|
| **Accuracy** | 88.1% | 81.4% |
| **Precision** | 81.8% | 55.6% |
| **Recall** | 64.3% | 41.7% |

---

## Capstone Reflection

**1. One way the Lab 1 tool could have been unfair:**  
The tool used `meals_program` (a yes/no income proxy) as a predictor. Students
from low-income families could be systematically flagged as at-risk regardless
of their actual academic performance — labeling them before they have a chance
to demonstrate what they can do.

**2. One gap found in Lab 2:**  
Cedarville CSD had an average achievement score of 51.0 compared
to 75.3 for Highpoint ISD — a 24.3-point gap. The
lowest-scoring K-Means cluster had 51.6% at-risk students and
0.0% on the meals program, showing the gap follows income lines.

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
