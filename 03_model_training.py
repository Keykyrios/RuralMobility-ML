"""
==============================================================================
Script 03 — Model Training, Cross-Validation & Hyperparameter Tuning
==============================================================================
An end-to-end Machine Learning pipeline for predicting Rural Trip Generation.

This script trains and compares multiple regression algorithms:

  1. DummyRegressor (mean baseline) — the "floor" every model must beat
  2. Linear Regression
  3. Ridge Regression (L2)
  4. Lasso Regression (L1)
  5. Decision Tree Regressor
  6. Random Forest Regressor
  7. Gradient Boosting Regressor
  8. XGBoost Regressor
  9. Support Vector Regression (SVR)

Each model is evaluated with 10-fold cross-validation.  The top 3 models
undergo hyperparameter tuning via GridSearchCV.  Results are visualised
with publication-quality comparison charts and exported to LaTeX tables.

==============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error
import xgboost as xgb

import config as cfg
import utils

# ─────────────────────────────────────────────────────────────────────────────
# 0.  LOAD PIPELINE ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("0.  LOADING PIPELINE ARTEFACTS")

artefacts = joblib.load(f"{cfg.MODELS_DIR}/pipeline_artefacts.joblib")

X_train = artefacts["X_train"]
y_train = artefacts["y_train"]
X_test = artefacts["X_test"]
y_test = artefacts["y_test"]
feature_names = artefacts["feature_names"]

print(f"  X_train: {X_train.shape}   y_train: {y_train.shape}")
print(f"  X_test : {X_test.shape}    y_test : {y_test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DEFINE ALL CANDIDATE MODELS
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("1.  CANDIDATE MODELS")

models = {
    "Dummy (Baseline)": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0, random_state=cfg.RANDOM_STATE),
    "Lasso Regression": Lasso(alpha=0.1, random_state=cfg.RANDOM_STATE,
                               max_iter=5000),
    "Decision Tree": DecisionTreeRegressor(random_state=cfg.RANDOM_STATE),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=cfg.RANDOM_STATE, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, random_state=cfg.RANDOM_STATE
    ),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=100, random_state=cfg.RANDOM_STATE,
        verbosity=0, n_jobs=-1
    ),
    "SVR": SVR(kernel="rbf", C=1.0),
}

for name in models:
    print(f"  • {name}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  10-FOLD CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("2.  10-FOLD CROSS-VALIDATION")

cv_results = {}

print(f"\n  {'Model':<25s} {'Mean RMSE':>12s} {'± Std':>10s} {'Time (s)':>10s}")
utils.print_divider()

for name, model in models.items():
    start = time.time()
    scores = cross_val_score(
        model, X_train, y_train,
        scoring="neg_mean_squared_error",
        cv=cfg.CV_FOLDS,
    )
    elapsed = time.time() - start
    rmse_scores = np.sqrt(-scores)
    cv_results[name] = {
        "rmse_mean": rmse_scores.mean(),
        "rmse_std": rmse_scores.std(),
        "rmse_scores": rmse_scores,
        "time_s": elapsed,
    }
    print(f"  {name:<25s} {rmse_scores.mean():>12.4f} {rmse_scores.std():>10.4f}"
          f" {elapsed:>10.2f}")

utils.print_divider()

# Create CV results DataFrame
cv_df = pd.DataFrame({
    "Model": list(cv_results.keys()),
    "Mean RMSE": [v["rmse_mean"] for v in cv_results.values()],
    "Std RMSE": [v["rmse_std"] for v in cv_results.values()],
    "Training Time (s)": [v["time_s"] for v in cv_results.values()],
}).sort_values("Mean RMSE").reset_index(drop=True)
cv_df.index = range(1, len(cv_df) + 1)
cv_df.index.name = "Rank"

print("\n  Cross-Validation Rankings:")
print(cv_df.to_string())

# LaTeX export
utils.save_table_to_latex(
    cv_df,
    filename="cross_validation_results",
    caption="10-Fold Cross-Validation Results — RMSE Comparison",
    label="tab:cv_results",
)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  MODEL COMPARISON — BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("3.  MODEL COMPARISON VISUALIZATION")

# Sort by Mean RMSE
sorted_models = sorted(cv_results.items(), key=lambda x: x[1]["rmse_mean"])
model_names_sorted = [m[0] for m in sorted_models]
means_sorted = [m[1]["rmse_mean"] for m in sorted_models]
stds_sorted = [m[1]["rmse_std"] for m in sorted_models]
colors_sorted = [cfg.MODEL_COLORS.get(m, "#888888") for m in model_names_sorted]

fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.barh(
    range(len(model_names_sorted)),
    means_sorted,
    xerr=stds_sorted,
    color=colors_sorted,
    edgecolor="white",
    linewidth=0.5,
    height=0.6,
    capsize=4,
    error_kw={"linewidth": 1, "capthick": 1, "ecolor": "#555"},
)
ax.set_yticks(range(len(model_names_sorted)))
ax.set_yticklabels(model_names_sorted, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Root Mean Squared Error (RMSE)", fontsize=11)
ax.set_title("Model Comparison — 10-Fold Cross-Validation RMSE (Lower = Better)",
             fontsize=13, fontweight="bold", pad=15)

# Annotate bars with values
for i, (mean, std) in enumerate(zip(means_sorted, stds_sorted)):
    ax.text(mean + std + 0.02, i, f"{mean:.3f} ± {std:.3f}",
            va="center", fontsize=9, color="#333")

# Mark the Dummy baseline with a dashed vertical line
dummy_rmse = cv_results["Dummy (Baseline)"]["rmse_mean"]
ax.axvline(x=dummy_rmse, color="#B0BEC5", linestyle="--", linewidth=1.5,
           label=f"Dummy Baseline = {dummy_rmse:.3f}")
ax.legend(loc="lower right", fontsize=9)

plt.tight_layout()
utils.save_figure("model_comparison_cv_rmse", sub_dir="model")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  CROSS-VALIDATION DISTRIBUTION — BOX PLOT
# ─────────────────────────────────────────────────────────────────────────────

utils.print_subsection("CV Score Distribution — Box Plot")

cv_data = []
for name, res in cv_results.items():
    for score in res["rmse_scores"]:
        cv_data.append({"Model": name, "RMSE": score})
cv_box_df = pd.DataFrame(cv_data)

# Order by median RMSE
order = cv_box_df.groupby("Model")["RMSE"].median().sort_values().index

fig, ax = plt.subplots(figsize=(12, 7))
sns.boxplot(
    data=cv_box_df, x="RMSE", y="Model", order=order,
    palette=[cfg.MODEL_COLORS.get(m, "#888") for m in order],
    linewidth=0.8, ax=ax,
    flierprops={"marker": "D", "markersize": 4, "alpha": 0.6},
)
ax.axvline(x=dummy_rmse, color="#B0BEC5", linestyle="--", linewidth=1.5,
           label="Dummy Baseline")
ax.set_title("Cross-Validation RMSE Distribution Across Models",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("RMSE (10 Folds)", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
utils.save_figure("cv_rmse_boxplot", sub_dir="model")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  HYPERPARAMETER TUNING — TOP 3 MODELS
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("5.  HYPERPARAMETER TUNING (GridSearchCV)")

# Identify top 3 non-dummy models
top3_names = [m for m, _ in sorted_models if m != "Dummy (Baseline)"][:3]
print(f"  Tuning top 3 models: {top3_names}")

param_grids = {
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    },
    "Decision Tree": {
        "max_depth": [3, 5, 10, 15, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
    },
    "Ridge Regression": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "Lasso Regression": {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
    },
    "Linear Regression": {},
    "SVR": {
        "C": [0.1, 1.0, 10.0],
        "epsilon": [0.01, 0.1, 0.5],
        "kernel": ["rbf", "linear"],
    },
}

tuned_models = {}

for name in top3_names:
    utils.print_subsection(f"Tuning: {name}")
    base_model = models[name]
    grid = param_grids.get(name, {})

    if not grid:
        print(f"    No hyperparameter grid defined — using default.")
        base_model.fit(X_train, y_train)
        tuned_models[name] = base_model
        continue

    grid_search = GridSearchCV(
        base_model, grid,
        cv=cfg.CV_FOLDS,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    start = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start

    best_rmse = np.sqrt(-grid_search.best_score_)
    print(f"    Best RMSE : {best_rmse:.4f}")
    print(f"    Best Params: {grid_search.best_params_}")
    print(f"    Tuning Time: {elapsed:.1f}s")

    tuned_models[name] = grid_search.best_estimator_

# Also train remaining models (un-tuned) for comparison
for name, model in models.items():
    if name not in tuned_models:
        model.fit(X_train, y_train)
        tuned_models[name] = model

# ─────────────────────────────────────────────────────────────────────────────
# 6.  LEARNING CURVE — BEST MODEL
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("6.  LEARNING CURVE — BEST MODEL")

best_model_name = top3_names[0]
best_model = tuned_models[best_model_name]

print(f"  Plotting learning curve for: {best_model_name}")

train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train, y_train,
    cv=cfg.CV_FOLDS,
    scoring="neg_mean_squared_error",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
)

train_rmse = np.sqrt(-train_scores)
val_rmse = np.sqrt(-val_scores)

fig, ax = plt.subplots(figsize=(10, 6))
ax.fill_between(train_sizes, train_rmse.mean(axis=1) - train_rmse.std(axis=1),
                train_rmse.mean(axis=1) + train_rmse.std(axis=1),
                alpha=0.15, color="#42A5F5")
ax.fill_between(train_sizes, val_rmse.mean(axis=1) - val_rmse.std(axis=1),
                val_rmse.mean(axis=1) + val_rmse.std(axis=1),
                alpha=0.15, color="#EF5350")
ax.plot(train_sizes, train_rmse.mean(axis=1), "o-", color="#42A5F5",
        label="Training RMSE", linewidth=2, markersize=5)
ax.plot(train_sizes, val_rmse.mean(axis=1), "o-", color="#EF5350",
        label="Validation RMSE", linewidth=2, markersize=5)
ax.set_xlabel("Number of Training Samples", fontsize=11)
ax.set_ylabel("RMSE", fontsize=11)
ax.set_title(f"Learning Curve — {best_model_name}",
             fontsize=13, fontweight="bold", pad=15)
ax.legend(fontsize=10, loc="upper right")
plt.tight_layout()
utils.save_figure("learning_curve_best_model", sub_dir="model")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  SAVE TRAINED MODELS
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("7.  SAVING TRAINED MODELS")
utils.create_output_dirs()

model_artefacts = {
    "tuned_models": tuned_models,
    "cv_results": cv_results,
    "best_model_name": best_model_name,
    "top3_names": top3_names,
}

path = joblib.dump(model_artefacts, f"{cfg.MODELS_DIR}/trained_models.joblib")[0]
print(f"  ✓ All trained models saved → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("MODEL TRAINING COMPLETE")
print(f"  Models trained     : {len(tuned_models)}")
print(f"  Best model (CV)    : {best_model_name}")
print(f"  Dummy Baseline RMSE: {cv_results['Dummy (Baseline)']['rmse_mean']:.4f}")
baseline = cv_results["Dummy (Baseline)"]["rmse_mean"]
best_rmse_val = cv_results[best_model_name]["rmse_mean"]
improvement = ((baseline - best_rmse_val) / baseline) * 100
print(f"  Best Model RMSE    : {best_rmse_val:.4f}")
print(f"  Improvement over   : {improvement:.1f}% better than predicting the mean")
print()
