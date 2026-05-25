# AIPL 2026 — IPL Match Outcome Forecasting

> **Competition:** AIPL 2026 IPL Match Forecast — Cleaned  
> **Task:** 4-class multiclass probability prediction  
> **Metric:** Multi-class Log Loss (lower is better)  
> **Final Submissions:** `submission_final_076.csv` · `submission_v8_soft_80.csv` (53 match rows each)

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Target Classes](#target-classes)
3. [Dataset Overview](#dataset-overview)
4. [Repository Structure](#repository-structure)
5. [Pipeline Architecture](#pipeline-architecture)
6. [Feature Engineering](#feature-engineering)
7. [Models & Ensembling](#models--ensembling)
8. [Probability Calibration](#probability-calibration)
9. [Submission Versioning](#submission-versioning)
10. [How to Reproduce](#how-to-reproduce)
11. [Dependencies](#dependencies)
12. [Key Design Decisions & Anti-Leakage Guarantees](#key-design-decisions--anti-leakage-guarantees)

---

## Problem Statement

Given pre-match and historical information about two IPL teams, predict the **probability distribution** across four possible match outcome categories. Predictions are evaluated using **multi-class log loss**, so well-calibrated probabilities matter far more than picking a single winner.

---

## Target Classes

The 4-class target is derived from historical ball-by-ball match data using the following logic:

| Class | Meaning | Derivation Rule |
|---|---|---|
| `A_small` | Bat-first team wins by a narrow margin | Winner == Bat First AND (Inn1 runs − Inn2 runs) ≤ 20 |
| `A_big` | Bat-first team wins by a large margin | Winner == Bat First AND (Inn1 runs − Inn2 runs) > 20 |
| `B_small` | Bat-second team wins with few wickets left | Winner == Bat Second AND wickets remaining < 6 |
| `B_big` | Bat-second team wins comfortably | Winner == Bat Second AND wickets remaining ≥ 6 |

Matches with `result_type` of `tie` or `no result` are excluded from training.

---

## Dataset Overview

| File | Description |
|---|---|
| `train_IPL.csv` | Ball-by-ball data for all historical IPL matches |
| `public_lb_matches.csv` | 2025–26 season matches where toss result is known |
| `schedule.csv` | Upcoming 2026 matches where toss is not yet known (private test set) |
| `sample_submission.csv` | Submission format template |
| `team_name_mapping.csv` | Canonical team name normalization table |

The training set is ball-by-ball granularity. All match-level features are constructed from this raw data without using any information that would not be available before the match begins.

---

## Repository Structure

```
.
├── aipl_submission.ipynb       # Full end-to-end pipeline notebook (all versions)
├── submission_final_076.csv    # Final selected submission 1 (Kaggle)
├── submission_v8_soft_80.csv   # Final selected submission 2 (Kaggle)
└── README.md
```

---

## Pipeline Architecture

The notebook is structured as two major versioned pipelines plus a post-processing stage.

### Stage 1 — Data Preparation (shared across versions)

**Step 1: Team name normalization.**  
All team names in the training set, public leaderboard file, and schedule are mapped through `team_name_mapping.csv` to a canonical form (e.g., "Delhi Daredevils" → "Delhi Capitals"). This prevents identity mismatch during historical lookups.

**Step 2: Match-level table construction.**  
The ball-by-ball `train_IPL.csv` is collapsed into one row per match. For each match, the last ball of each innings is used to capture the final score, wickets, and balls remaining for both innings.

```
build_match_table(df)
  → groups by Match ID
  → pivots last-ball stats per innings
  → merges with legal ball counts
```

**Step 3: 4-class label derivation.**  
Labels are derived using the margin-based rules described in [Target Classes](#target-classes). Ties and no-results are excluded. This yields a clean labelable match set sorted chronologically.

**Step 4: Team history table construction.**  
For each match-innings pair, a row is written per team (batting perspective + bowling perspective). This produces a long-format history table that is queried at feature-building time using only records dated strictly before the current match.

---

### Stage 2 — Feature Engineering

See [Feature Engineering](#feature-engineering) section below.

---

### Stage 3 — Data Augmentation

To eliminate the artificial asymmetry where "Team A" always refers to the batting-first team in training data, every match is represented **twice**:

- **Original perspective:** Team A = Bat First, Team B = Bat Second
- **Flipped perspective:** Team A = Bat Second, Team B = Bat First, label flipped (A↔B side swap)

This forces the model to be invariant to which team is listed first, producing a more generalizable classifier.

---

### Stage 4 — Validation

A **time-based split** is used: all matches from 2025 onward are held out as a validation set. If this produces fewer than 80 validation samples, the cutoff shifts to the 82nd percentile date. This strictly prevents temporal data leakage that would occur with random k-fold splitting on sequential sports data.

---

### Stage 5 — Model Training & Inference

See [Models & Ensembling](#models--ensembling).

---

### Stage 6 — Prediction for Two Submission Types

The competition test set has two types of rows:

**Public LB matches (toss is known):** Toss winner and decision are available in `public_lb_matches.csv`. A single feature row is built and scored directly.

**Private schedule matches (toss is unknown):** Toss has not happened yet. Four scenarios are enumerated (Team A wins toss and bats; Team A wins toss and fields; Team B wins toss and bats; Team B wins toss and fields), each with probability weight derived from the venue's historical toss-decision rate. The four resulting probability vectors are averaged by their scenario weights — this is a **marginalization over toss uncertainty**.

---

## Feature Engineering

All features are constructed using only data from matches that occurred **strictly before** the date of the match being predicted.

### Team Rolling Statistics

For each team (computed separately for Team A and Team B, then differenced):

| Feature | Description |
|---|---|
| `win_rate` | Fraction of recent matches won |
| `bat_run_rate` | Runs per legal ball faced (batting) |
| `bat_sr` | Batter strike rate (batter runs × 100 / legal balls) |
| `boundary_rate` | Boundaries per legal ball |
| `dot_rate` | Dot balls per legal ball |
| `avg_wickets_lost` | Average wickets lost per match |
| `bowl_economy` | Bowling economy (runs per over conceded) |
| `bowl_wicket_rate` | Wickets taken per legal ball bowled |

**V1** uses a single 5-match rolling window.  
**V2** computes the above across three windows: last 5, last 10, and last 20 matches.

### Phase-Level Statistics (V2 only)

Each innings is broken into three phases. Batting and bowling metrics are tracked separately for each:

| Phase | Overs |
|---|---|
| Powerplay | 1–6 |
| Middle overs | 7–15 |
| Death overs | 16–20 |

For each phase: run rate, dot rate, boundary rate, wickets.

### Margin History Features (V2 only)

| Feature | Description |
|---|---|
| `big_win_rate` | Rate of matches won by a large margin |
| `small_win_rate` | Rate of matches won by a narrow margin |
| `big_loss_rate` | Rate of matches lost by a large margin |
| `small_loss_rate` | Rate of matches lost by a narrow margin |

### Venue Features

Computed from all prior matches at the same venue (fallback to same city, then global if insufficient data):

| Feature | Description |
|---|---|
| `venue_avg_inn1_runs` | Average first-innings total |
| `venue_avg_inn2_runs` | Average second-innings total |
| `venue_chase_win_rate` | Historical chase success rate |
| `venue_bat_first_win_rate` | Historical bat-first win rate |
| `venue_toss_bat_rate` | Rate of toss winners choosing to bat |
| `venue_big_rate` / `venue_small_rate` | Historical margin distribution (V2) |
| `venue_A/B_big/small_rate` | Per-class historical frequencies (V2) |

### Head-to-Head Features

Derived from the last 10 meetings between the two teams:

| Feature | Description |
|---|---|
| `h2h_count` | Number of prior H2H matches |
| `h2h_A_win_rate` | Team A historical win rate in H2H matchups |
| `h2h_big_rate` / `h2h_small_rate` | Margin distribution in H2H history (V2) |

### Toss & Contextual Features

| Feature | Description |
|---|---|
| `toss_winner_is_A` | Binary: did Team A win the toss (1) or Team B (0) |
| `toss_decision_bat` | Binary: did the toss winner elect to bat (1) or field (0) |
| `year`, `month`, `dayofyear` | Temporal features |

### Difference Features

For every rolling/phase statistic computed for both Team A and Team B, a difference feature (`A_metric − B_metric`) is added to capture the relative advantage.

---

## Models & Ensembling

### V1 — Three-Model Ensemble

Three gradient boosting classifiers are trained with a `multiclass` objective (4 classes):

**LightGBM**
```
n_estimators=700, learning_rate=0.025, max_depth=3,
num_leaves=15, subsample=0.85, colsample_bytree=0.75,
reg_alpha=0.8, reg_lambda=3.0
```

**XGBoost**
```
n_estimators=600, learning_rate=0.025, max_depth=3,
subsample=0.85, colsample_bytree=0.75,
reg_alpha=0.5, reg_lambda=3.0, tree_method=hist
```

**CatBoost**
```
iterations=800, depth=4, learning_rate=0.025, l2_leaf_reg=6
```

Categorical columns (team names, venue, city) are one-hot encoded for LightGBM and XGBoost, and passed as native categoricals for CatBoost.

Final ensemble is a weighted average searched over 7 weight combinations during validation.

---

### V2 — CatBoost Solo with Native Categoricals

Uses the enriched V2 feature set (phase stats, multi-window rolling stats, margin history). CatBoost is used exclusively due to its superior handling of categorical features natively:

```
iterations=1200, depth=4, learning_rate=0.018,
l2_leaf_reg=8, random_strength=1.5,
bagging_temperature=0.7, border_count=128
```

Early stopping is applied against the validation set. Final model is retrained for `best_iteration + 1` iterations on all data.

---

## Probability Calibration

After getting raw predicted probabilities, two calibration steps are applied:

**1. Gamma (Temperature) Scaling**

```python
p = p ** gamma
p = p / p.sum(axis=1, keepdims=True)
```

`gamma < 1` softens the distribution toward uniform (reduces overconfidence).  
`gamma > 1` sharpens the distribution (increases confidence).  
Optimal gamma is grid-searched over [0.60, 1.30] against the validation log loss.

**2. Prior Blending**

```python
p = alpha * p + (1 - alpha) * prior
```

Blends the model's prediction with the training prior. This regularizes predictions toward the base rate, which is effective when the model makes very extreme predictions on matches with limited historical data.  
Optimal alpha is grid-searched over [0.55, 1.00].

**Post-processing versions (v8–v15)** additionally apply:
- Hard probability clipping (e.g., clip to [0.17, 0.36]) to prevent extreme log loss on any single row
- Targeted class-level boosts/reductions to correct systematic imbalances (e.g., `B_big` was historically over-predicted)
- Uncertainty-based diversity enforcement (v15): the 12 most uncertain rows are nudged toward `A_big` to improve ranking spread

---

## Submission Versioning

| File | Core Model | Feature Set | Calibration | Notes |
|---|---|---|---|---|
| v5 (intermediate) | LightGBM + XGBoost + CatBoost | V1 rolling (5-match window) | Grid-searched gamma + prior blend | Baseline ensemble |
| v6 (intermediate) | CatBoost V2 | Phase stats + multi-window (5/10/20) + margin history | Grid-searched gamma + prior blend | Richer feature set |
| **`submission_v8_soft_80.csv`** ✅ | Post-processing on v5/v6 | — | Soft temperature (T=0.80) + hard clip | **Final Kaggle selection** |
| v12–v13 (intermediate) | Ensemble of v5, v6, v8 | — | Diversity-corrected | Class balance tuning |
| v14 (intermediate) | Ensemble (0.42×v6 + 0.33×v5 + 0.25×v8) | — | Elite calibration (clip [0.18, 0.355]) | Best ensemble attempt |
| **`submission_final_076.csv`** ✅ | Ensemble + post-processing | Phase + multi-window + margin history | Full calibration pipeline (γ, prior blend, clip) | **Final Kaggle selection** |

---

## How to Reproduce

### Environment

Google Colab (GPU not required; all models run on CPU).

### Steps

1. Upload `chart_aipl-2026-ipl-match-forecast-cleaned.zip` to the Colab session.
2. Open `aipl_submission.ipynb` and run all cells in order.
3. The notebook will:
   - Unzip data and locate `train_IPL.csv`
   - Build the match table and team history
   - Derive labels and construct the augmented training set
   - Train and validate all models
   - Generate predictions for all 53 test rows
   - Save submission files

> **Important:** Run cells sequentially. V2 cells depend on the `labelable_matches` and `team_history_v2` objects built in earlier cells.

---

## Dependencies

```
lightgbm
xgboost
catboost
scikit-learn
numpy
pandas
```

Install in Colab:
```bash
!pip install lightgbm xgboost catboost -q
```

---

## Key Design Decisions & Anti-Leakage Guarantees

**No ball-by-ball test-time data.** All feature engineering uses only pre-match history (team form, venue trends, H2H). Ball-by-ball data from the test match itself is never accessed.

**Strict temporal filtering.** Every call to `summarize_team`, `summarize_venue`, or `summarize_h2h` filters the history to `Date < current_match_date`. There is no row in any feature vector that uses information from the same match or any future match.

**Time-based validation, not k-fold.** Random splitting would leak future match patterns into training. The validation set always represents the most recent matches in the dataset.

**Symmetric augmentation.** Without flipping, the model would learn a spurious signal that "Team A" (always batting first in training) is stronger. The flip augmentation removes this bias entirely.

**Toss marginalization for private matches.** Because the toss for future matches is unknown at prediction time, the model does not receive toss information for those rows. Instead, the four possible toss scenarios are averaged with venue-calibrated weights — this is a proper marginalization rather than a point guess.

**Probability clipping.** Log loss penalizes extreme predictions severely. Hard clipping to a minimum of ~0.17 per class prevents catastrophic loss on any single match row, at the cost of slight accuracy on confident predictions.

---

*This README reflects the state of the project as submitted for the AIPL 2026 IPL Match Forecasting competition.*
