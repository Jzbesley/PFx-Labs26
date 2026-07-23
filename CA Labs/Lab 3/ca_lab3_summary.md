# California Modeling Equity — Full Series Capstone Summary
**Generated:** July 23, 2026  
**State:** California (synthetic data — no real student records used)

---

## Lab 1 — Prediction Tool

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| **Accuracy** | 83.0% | 81.4% |
| **Precision** | 80.0% | 100.0% |
| **Recall** | 30.8% | 15.4% |

---

## Lab 2 — Where the Gaps Show Up

### K-Means Clusters (K=4)

| Cluster | Students | Avg Score | At-Risk % | Meals % |
|---|---|---|---|---|
| 3 | 63 | 61.9 | 0.0% | 0.0% |
| 0 | 69 | 61.1 | 0.0% | 0.0% |
| 1 | 67 | 59.6 | 0.0% | 0.0% |
| 2 | 92 | 50.1 | 0.0% | 0.0% |


### District Achievement Scores

| District | Avg Score |
|---|---|
| San Jose USD | 72.5 |
| San Francisco USD | 65.1 |
| San Diego USD | 62.3 |
| Los Angeles USD | 56.2 |
| Sacramento City USD | 56.0 |
| Oakland USD | 54.4 |
| Fresno USD | 48.2 |
| Bakersfield City SD | 46.1 |


**Lowest district:** Bakersfield City SD (avg: 46.1)  
**Highest district:** San Jose USD (avg: 72.5)  
**Gap:** 26.4 points

**Top feature (mutual information):** `avg_quiz_score` (MI=0.0983)  
**Top permutation importance feature:** `meals_program` (+97.95 MSE)

---

## Lab 3 — Fixing the Data

| | Warnings |
|---|---|
| Original data | 2 |
| Redesigned data | 1 |
| Improvement | 1 warning(s) removed |

### Model Performance: Original vs. Redesigned

| Metric | Original | Redesigned |
|---|---|---|
| **Accuracy** | 83.0% | 76.3% |
| **Precision** | 80.0% | 75.0% |
| **Recall** | 30.8% | 18.8% |

---

## Capstone Reflection

**One unfairness in Lab 1:** Using `meals_program` as a binary predictor
means the model treats income as a proxy for academic risk, potentially
over-flagging low-income students in districts like Bakersfield City SD.

**One gap found in Lab 2:** A 26.4-point achievement gap exists between
Bakersfield City SD and San Jose USD, mirroring California's well-documented
resource disparities between wealthy and under-resourced districts.

**One problem fixed in Lab 3:** `meals_program` (binary) was replaced with
`economic_need_level` (4 levels), reducing equity warnings from 2 to 1.

**Advice for a school:** Treat the model as one signal — not a decision.
Audit regularly for disparate impact, especially across district lines.

---
*All data is synthetic — no real student records were used.*
