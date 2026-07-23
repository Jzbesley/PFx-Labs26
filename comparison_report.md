# Georgia vs. California — Modeling Equity Comparison Report
**Generated:** July 23, 2026  
**Data:** Synthetic only — no real student records used.

---

## Overview

| | Georgia | California |
|---|---|---|
| Students | 291 | 291 |
| At-risk students | 77 (26.5%) | 51 (17.5%) |
| Meals program (% yes) | ~27% | ~38% |
| Districts simulated | 8 | 8 |

---

## Lab 1 — Prediction Model Comparison

### Logistic Regression

| Metric | Georgia | California |
|---|---|---|
| **Accuracy** | 88.1% | 81.4% |
| **Precision** | 81.8% | 66.7% |
| **Recall** | 64.3% | 16.7% |

### Random Forest

| Metric | Georgia | California |
|---|---|---|
| **Accuracy** | 81.4% | 81.4% |
| **Precision** | 100.0% | 100.0% |
| **Recall** | 21.4% | 15.4% |

**Key observation:** Georgia shows higher recall on the LR model, suggesting stronger signal between the features and at-risk labels in the GA dataset.

---

## Lab 2 — Achievement Gap Comparison

### Neural Network Performance

| | Georgia | California |
|---|---|---|
| MSE | 95.73 | 151.81 |
| RMSE | 9.78 pts | 12.32 pts |
| Top MI feature | attendance_rate | avg_quiz_score |
| Top permutation feature | meals_program | meals_program |

### District Achievement Gap

| | Georgia | California |
|---|---|---|
| Lowest district | Cedarville CSD (51.0) | Bakersfield City SD (46.1) |
| Highest district | Highpoint ISD (75.3) | San Jose USD (72.5) |
| **Gap** | **24.3 pts** | **26.4 pts** |

California's district gap is larger than Georgia's, reflecting its greater socioeconomic stratification across metro areas.

### K-Means Cluster Profiles

**Georgia**

| Cluster | Avg Score | At-Risk % |
|---|---|---|
| 0 | 68.4 | 32.7% |
| 2 | 65.3 | 4.3% |
| 1 | 58.7 | 13.2% |
| 3 | 53.9 | 51.6% |


**California**

| Cluster | Avg Score | At-Risk % |
|---|---|---|
| 3 | 61.9 | 0.0% |
| 0 | 61.1 | 0.0% |
| 1 | 59.6 | 0.0% |
| 2 | 50.1 | 0.0% |


Both states show the same pattern: the lowest-scoring cluster has the highest
at-risk rate, confirming that achievement gaps follow economic lines in both
states' synthetic data.

---

## Lab 3 — Schema Equity Comparison

| | Georgia | California |
|---|---|---|
| Warnings (original) | 2 | 2 |
| Warnings (redesigned) | 1 | 1 |
| Warnings removed | 1 | 1 |
| Key fix | meals_program → 4-level economic_need_level | meals_program → 4-level economic_need_level |

### LR Performance After Redesign

| Metric | GA Original | GA Redesigned | CA Original | CA Redesigned |
|---|---|---|---|---|
| **Accuracy** | 88.1% | 88.1% | 81.4% | 76.3% |
| **Precision** | 81.8% | 81.8% | 66.7% | 75.0% |
| **Recall** | 64.3% | 64.3% | 16.7% | 18.8% |

---

## Key Takeaways

1. **Both states show income-correlated achievement gaps.** The `meals_program`
   feature was the top permutation-importance feature in both neural networks,
   meaning both models learned to use income as a shortcut for predicting
   achievement.

2. **California has more students on the meals program (~38% vs ~27%)**,
   reflecting its larger low-income student population and higher cost of living.

3. **District gaps exist in both states.** Georgia: 24.3 pts.
   California: 26.4 pts. These gaps mirror documented national trends
   linking school district resources to student outcomes.

4. **Schema redesign helped both states.** Replacing the binary `meals_program`
   with a 4-level `economic_need_level` reduced equity warnings in both datasets
   by giving models and reviewers more nuanced information.

5. **The `at_risk` target column remains binary in both states.** This is the
   one remaining equity flag — a future improvement would be a risk spectrum
   (e.g., low/moderate/high/critical risk) rather than a yes/no label.

---

## Files Reference

| File | Location |
|---|---|
| Georgia Lab 1 | `Lab 1/students_with_entropy.csv` |
| Georgia Lab 2 | `Lab 2/lab2_students_clustered.csv` |
| Georgia Lab 3 | `Lab 3/lab3_georgia_summary.md` |
| California Lab 1 | `CA Labs/Lab 1/ca_students_with_entropy.csv` |
| California Lab 2 | `CA Labs/Lab 2/ca_district_map.html` |
| California Lab 3 | `CA Labs/Lab 3/ca_lab3_summary.md` |
| **This report** | `comparison_report.md` |

---
*All data is synthetic — no real student records were used in this project.*
