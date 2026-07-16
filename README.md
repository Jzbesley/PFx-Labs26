# Modeling Equity — Lab Series

A three-day, hands-on coding series for high school seniors. Students use AI coding tools to explore real research on fairness in education — no prior coding experience required.

Each lab runs **1:30 – 4:00 PM (2.5 hours)** and builds directly on the one before it, so work must be saved and carried forward each day.

---

## What Is "Vibe Coding"?

Vibe coding means describing what you want to build in plain English and working with an AI assistant step by step — instead of writing every line of code yourself. You stay in charge: you decide what to build, check whether the output makes sense, and speak up when something looks off.

Supported tools:
- **Kiro AI** (this IDE)
- Google Gemini inside Google Colab
- Any other AI coding tool approved by your instructor

---

## Lab Overview

| Lab | Based On | What You'll Build | Take-Away |
|-----|----------|-------------------|-----------|
| 1 | *Predictive Analytics for Educational Equity in Low-Resource Schools* | A tool that predicts which students might need extra help | Working prediction tool + attendance entropy score |
| 2 | *Exploring Educational Equity and Achievement Disparities in Georgia* | Analysis of where and for whom the tool works differently | Short report + district achievement map |
| 3 | *Building Equitable Education Datasets for Developing Nations* | A checker that finds hidden bias in the data itself | Fairness-checker tool + written summary |

---

## Repository Structure

```
PFx Labs/
├── Lab 1/          # Prediction tool, cleaned data, entropy scores
├── Lab 2/          # Feature rankings, neural network, clusters, map
├── Lab 3/          # Redesigned schema, check_schema_equity() tool
├── .gitignore
└── README.md
```

---

## Lab 1 — Teaching a Computer to Spot Warning Signs

**Goal:** Build a logistic regression and random forest model to predict student risk, then calculate an attendance entropy score.

Key steps:
1. Generate a 300-row synthetic student dataset
2. Fill missing `avg_quiz_score` values using median imputation
3. Train a Logistic Regression model (accuracy, precision, recall)
4. Train a Random Forest and compare results
5. Calculate `attendance_entropy` using H = −Σ pᵢ log(pᵢ)
6. Reflect on the fairness implications of using `meals_program` as a feature

**Turn in:** Cleaned dataset, both models with scores, `attendance_entropy` column, written reflection.

---

## Lab 2 — Where Do the Gaps Show Up?

**Goal:** Extend Lab 1 data with district info and an achievement score, then find where the model performs differently across groups.

Key steps:
1. Add `district`, `achievement_score`, and lat/lon columns
2. Rank features with `SelectKBest` (mutual information)
3. Standardize numeric columns and train an `MLPRegressor`
4. Group students into 4 clusters with K-Means
5. Run permutation importance and produce a bar chart
6. Build an interactive map (folium or plotly) of average scores by district
7. Reflect on overlap between low-scoring clusters and Lab 1 at-risk flags

**Turn in:** Ranked features, neural network + MSE score, cluster averages, bar chart, map, written reflection.

---

## Lab 3 — Fixing the Data Before It Causes Harm

**Goal:** Audit the dataset for structural bias, redesign overly coarse columns, and build a reusable schema-equity checker.

Key steps:
1. Categorize every column (binary / few categories / numeric range) and flag overly simple ones
2. Discuss SDMX (Statistical Data and Metadata eXchange) and why consistent categories matter
3. Replace `meals_program` (yes/no) with `economic_need_level` (4+ levels)
4. Add `sub_region` alongside `district`
5. Build `check_schema_equity(dataframe)` — warns on binary or under-categorized sensitive columns
6. Run the checker on original vs. redesigned data and compare warning counts
7. Write an 8–10 sentence capstone summary covering all three labs

**Turn in:** Bias-check comparison table, redesigned columns, working `check_schema_equity()` function with before/after output, capstone summary.

---

## Privacy Notice

**No real student records are used.** Every dataset is synthetically generated to resemble the patterns described in the source research papers. Students practice safely without accessing anyone's private information.

---

## Grading Rubric

| Component | Weight |
|-----------|--------|
| Completed work turned in (each lab) | 40% |
| Written answers / reflections | 30% |
| Daily quiz score (10 questions, Google Forms) | 30% |

---

## Troubleshooting Quick Reference

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run `!pip install <package-name>` in a Colab cell, then re-import |
| Shape mismatch | Ask the AI to print `.shape` on each array before the failing line |
| Colab freezes | Runtime → Restart runtime, then re-run cells from the top |
| Results look wrong | Ask the AI to print the first few rows of any new column right after creating it |

---

## Tips for Getting Good Results from Your AI Assistant

1. Say what you have and what you want: *"I have columns X, Y, Z — I want a tool that predicts Y from X and Z."*
2. Ask for explanations before running code: *"Explain what this does before I run it."*
3. Work in small steps rather than asking for everything at once.
4. When you get an error, copy the exact message and ask the assistant to fix and explain it.
5. Ask the assistant to print results along the way so you can check each step.
