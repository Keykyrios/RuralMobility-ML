"""
==============================================================================
Script 04 — Model Evaluation & Diagnostic Analysis
==============================================================================
Author: Mitrajit Ghorui (Keykyrios)
GitHub: https://github.com/Keykyrios/RuralMobility-ML
==============================================================================
An end-to-end Machine Learning pipeline for predicting Rural Trip Generation.

Comprehensive model evaluation on the held-out test set:

  1. Test-set metrics for all trained models  (R², Adj. R², MAE, RMSE, MAPE)
  2. Actual vs Predicted scatter plot with confidence intervals
  3. Residual analysis  (scatter, histogram + KDE, Q-Q plot)
  4. Feature importance  (Random Forest permutation + SHAP analysis)
  5. Prediction error cumulative distribution
  6. Full metrics comparison table exported to LaTeX

==============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy import stats

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    explained_variance_score, mean_absolute_percentage_error,
)
from sklearn.inspection import permutation_importance

import config as cfg
import utils

# ─────────────────────────────────────────────────────────────────────────────
# 0.  LOAD ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("0.  LOADING ARTEFACTS")

pipe_art = joblib.load(f"{cfg.MODELS_DIR}/pipeline_artefacts.joblib")
model_art = joblib.load(f"{cfg.MODELS_DIR}/trained_models.joblib")

X_train = pipe_art["X_train"]
X_test = pipe_art["X_test"]
y_train = pipe_art["y_train"]
y_test = pipe_art["y_test"]
feature_names = pipe_art["feature_names"]

tuned_models = model_art["tuned_models"]
best_model_name = model_art["best_model_name"]
best_model = tuned_models[best_model_name]

print(f"  Best model : {best_model_name}")
print(f"  Test size  : {len(y_test)} samples")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  TEST-SET METRICS — ALL MODELS
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("1.  TEST-SET EVALUATION — ALL MODELS")

n = len(y_test)
p = X_test.shape[1]

metrics_rows = []

for name, model in tuned_models.items():
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    evs = explained_variance_score(y_test, preds)
    try:
        mape = mean_absolute_percentage_error(y_test, preds) * 100
    except Exception:
        mape = np.nan

    metrics_rows.append({
        "Model": name,
        "R²": r2,
        "Adj. R²": adj_r2,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "Expl. Var.": evs,
    })

metrics_df = pd.DataFrame(metrics_rows).sort_values("RMSE").reset_index(drop=True)
metrics_df.index = range(1, len(metrics_df) + 1)
metrics_df.index.name = "Rank"

print(metrics_df.to_string())

# LaTeX export
utils.save_table_to_latex(
    metrics_df.set_index("Model"),
    filename="test_set_metrics",
    caption="Test-Set Performance Metrics — All Regression Models",
    label="tab:test_metrics",
)

# Save as CSV too
metrics_df.to_csv(f"{cfg.RESULTS_DIR}/model_metrics_summary.csv", index=False)
print(f"\n  ✓ Metrics saved → {cfg.RESULTS_DIR}/model_metrics_summary.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  ACTUAL vs PREDICTED — BEST MODEL
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("2.  ACTUAL vs PREDICTED — SCATTER PLOT")

y_pred = best_model.predict(X_test)
r2_val = r2_score(y_test, y_pred)
rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(y_test, y_pred, alpha=0.55, s=45, color="#5C6BC0",
           edgecolors="white", linewidth=0.4, zorder=3)

# Perfect prediction line
lims = [
    min(y_test.min(), y_pred.min()) - 0.5,
    max(y_test.max(), y_pred.max()) + 0.5,
]
ax.plot(lims, lims, "--", color="#EF5350", linewidth=2, label="Perfect Prediction",
        zorder=2)

# Fit a regression line to show trend
slope, intercept, r, p_val, se = stats.linregress(y_test, y_pred)
x_line = np.linspace(lims[0], lims[1], 100)
ax.plot(x_line, slope * x_line + intercept, "-", color="#66BB6A",
        linewidth=1.5, alpha=0.7, label=f"Best Fit (slope={slope:.2f})")

# Annotate R² and RMSE
textstr = f"R² = {r2_val:.4f}\nRMSE = {rmse_val:.4f}"
ax.text(0.05, 0.92, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#ccc", alpha=0.9))

ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect("equal")
ax.set_xlabel("Actual Trips per Day (PCU)", fontsize=11)
ax.set_ylabel("Predicted Trips per Day (PCU)", fontsize=11)
ax.set_title(f"Actual vs Predicted — {best_model_name}",
             fontsize=13, fontweight="bold", pad=15)
ax.legend(fontsize=10, loc="lower right")
plt.tight_layout()
utils.save_figure("actual_vs_predicted", sub_dir="evaluation")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  RESIDUAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("3.  RESIDUAL ANALYSIS")

residuals = y_test.values - y_pred

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# (a) Residuals vs Predicted
axes[0].scatter(y_pred, residuals, alpha=0.5, s=40, color="#5C6BC0",
                edgecolors="white", linewidth=0.4)
axes[0].axhline(y=0, color="#EF5350", linestyle="--", linewidth=1.5)
axes[0].set_xlabel("Predicted Trips (PCU)")
axes[0].set_ylabel("Residual (Actual − Predicted)")
axes[0].set_title("(a) Residuals vs Predicted Values", fontweight="bold")

# (b) Residual distribution + KDE + Normal overlay
axes[1].hist(residuals, bins=20, density=True, color="#66BB6A",
             edgecolor="white", alpha=0.7, linewidth=0.5)
kde_x = np.linspace(residuals.min(), residuals.max(), 200)
kde = stats.gaussian_kde(residuals)
axes[1].plot(kde_x, kde(kde_x), color="#333", linewidth=2, label="KDE")
# Normal distribution overlay
mu, sigma = residuals.mean(), residuals.std()
normal_y = stats.norm.pdf(kde_x, mu, sigma)
axes[1].plot(kde_x, normal_y, "--", color="#EF5350", linewidth=1.5,
             label=f"Normal (μ={mu:.2f}, σ={sigma:.2f})")
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Density")
axes[1].set_title("(b) Residual Distribution", fontweight="bold")
axes[1].legend(fontsize=8)

# (c) Q-Q Plot
stats.probplot(residuals, dist="norm", plot=axes[2])
axes[2].set_title("(c) Q-Q Plot — Residual Normality", fontweight="bold")
axes[2].get_lines()[0].set(color="#5C6BC0", markersize=4, alpha=0.6)
axes[2].get_lines()[1].set(color="#EF5350", linewidth=1.5)

fig.suptitle(f"Residual Diagnostic Analysis — {best_model_name}",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
utils.save_figure("residual_analysis", sub_dir="evaluation")

# Shapiro-Wilk normality test for residuals
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"  Shapiro-Wilk Test  : W={shapiro_stat:.4f}, p={shapiro_p:.4f}")
if shapiro_p > 0.05:
    print("  → Residuals appear normally distributed (p > 0.05).")
else:
    print("  → Residuals deviate from normality (p ≤ 0.05).")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  FEATURE IMPORTANCE — PERMUTATION IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("4.  FEATURE IMPORTANCE — PERMUTATION IMPORTANCE")

perm_imp = permutation_importance(
    best_model, X_test, y_test,
    n_repeats=15,
    random_state=cfg.RANDOM_STATE,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
)

perm_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": perm_imp.importances_mean,
    "Std": perm_imp.importances_std,
}).sort_values("Importance", ascending=False)

top_n = 15
perm_top = perm_df.head(top_n).sort_values("Importance")

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(
    range(len(perm_top)),
    perm_top["Importance"],
    xerr=perm_top["Std"],
    color=sns.color_palette("muted", top_n)[::-1],
    edgecolor="white",
    linewidth=0.5,
    height=0.6,
    capsize=3,
)
ax.set_yticks(range(len(perm_top)))
ax.set_yticklabels(perm_top["Feature"], fontsize=9)
ax.set_xlabel("Mean Importance (Permutation)", fontsize=11)
ax.set_title(f"Top {top_n} Feature Importances — {best_model_name} (Permutation)",
             fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
utils.save_figure("feature_importance_permutation", sub_dir="evaluation")

# LaTeX export of feature importances
perm_latex = perm_df.head(top_n).reset_index(drop=True)
perm_latex.index = range(1, len(perm_latex) + 1)
perm_latex.index.name = "Rank"
utils.save_table_to_latex(
    perm_latex,
    filename="feature_importance",
    caption=f"Top {top_n} Feature Importances — Permutation Method",
    label="tab:feature_importance",
)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  SHAP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("5.  SHAP ANALYSIS — GLOBAL FEATURE IMPORTANCE")

try:
    import shap

    # Use a background sample for efficiency
    background = shap.sample(pd.DataFrame(X_train, columns=feature_names), 50)
    explainer = shap.Explainer(best_model.predict, background)
    shap_values = explainer(pd.DataFrame(X_test, columns=feature_names))

    # (a) SHAP Beeswarm summary
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title(f"SHAP Feature Importance — {best_model_name}",
              fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    utils.save_figure("shap_beeswarm", sub_dir="evaluation")

    # (b) SHAP Bar plot
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title(f"SHAP Mean |Impact| on Prediction — {best_model_name}",
              fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    utils.save_figure("shap_bar", sub_dir="evaluation")

    # (c) SHAP Dependence plots for top 3 features
    top3_shap_feats = (
        pd.DataFrame(np.abs(shap_values.values), columns=feature_names)
        .mean()
        .nlargest(3)
        .index.tolist()
    )

    for feat in top3_shap_feats:
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.scatter(shap_values[:, feat], show=False)
        plt.title(f"SHAP Dependence — {feat}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        safe_feat = feat.replace(" ", "_").replace("(", "").replace(")", "").lower()
        utils.save_figure(f"shap_dependence_{safe_feat}", sub_dir="evaluation")

    print("  ✓ SHAP analysis complete — beeswarm, bar, and dependence plots saved.")

except Exception as e:
    print(f"  ⚠ SHAP analysis skipped: {e}")
    print("    (This is non-critical — all other evaluations are complete.)")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  PREDICTION ERROR CUMULATIVE DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("6.  PREDICTION ERROR DISTRIBUTION")

abs_errors = np.abs(residuals)
sorted_errors = np.sort(abs_errors)
cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sorted_errors, cumulative, color="#5C6BC0", linewidth=2)
ax.fill_between(sorted_errors, cumulative, alpha=0.15, color="#5C6BC0")

# Mark key thresholds
for threshold, color, ls in [(1.0, "#66BB6A", "--"), (1.5, "#FFA726", "--"),
                               (2.0, "#EF5350", "--")]:
    pct = (abs_errors < threshold).sum() / len(abs_errors) * 100
    ax.axvline(x=threshold, color=color, linestyle=ls, linewidth=1.2,
               label=f"{pct:.0f}% within ±{threshold} PCU")

ax.set_xlabel("Absolute Prediction Error (PCU)", fontsize=11)
ax.set_ylabel("Cumulative Percentage (%)", fontsize=11)
ax.set_title(f"Cumulative Error Distribution — {best_model_name}",
             fontsize=13, fontweight="bold", pad=15)
ax.legend(fontsize=9, loc="lower right")
ax.set_ylim(0, 105)
plt.tight_layout()
utils.save_figure("cumulative_error_distribution", sub_dir="evaluation")

# Custom accuracy metrics
for t in [0.5, 1.0, 1.5, 2.0]:
    pct = (abs_errors < t).sum() / len(abs_errors) * 100
    print(f"  Predictions within ±{t} PCU : {pct:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  MODEL COMPARISON — TEST SET RADAR CHART
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("7.  MODEL COMPARISON — MULTI-METRIC BAR CHART")

# Filter out Dummy for multi-metric comparison
comparison_models = [m for m in metrics_df["Model"] if m != "Dummy (Baseline)"]
comp_df = metrics_df[metrics_df["Model"].isin(comparison_models)].set_index("Model")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, metric in enumerate(["R²", "MAE", "RMSE"]):
    data = comp_df[metric].sort_values(ascending=(metric != "R²"))
    colors = [cfg.MODEL_COLORS.get(m, "#888") for m in data.index]
    axes[i].barh(data.index, data.values, color=colors,
                 edgecolor="white", linewidth=0.5, height=0.6)
    axes[i].set_xlabel(metric, fontsize=11)
    axes[i].set_title(f"Test-Set {metric}", fontweight="bold")

    # Annotate values
    for j, (idx, val) in enumerate(data.items()):
        axes[i].text(val + 0.01 * data.max(), j, f"{val:.3f}",
                     va="center", fontsize=9, color="#333")

fig.suptitle("Multi-Metric Test-Set Comparison (Excluding Dummy Baseline)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
utils.save_figure("multi_metric_comparison", sub_dir="evaluation")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("MODEL EVALUATION COMPLETE")
print(f"  Best Model         : {best_model_name}")
print(f"  Test R²            : {r2_val:.4f}")
print(f"  Test RMSE          : {rmse_val:.4f}")
print(f"  Figures saved to   : {cfg.EVAL_FIGURES_DIR}")
print(f"  LaTeX tables       : {cfg.LATEX_DIR}")
print()
