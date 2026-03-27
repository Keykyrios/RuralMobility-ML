"""
==============================================================================
Script 01 — Data Loading & Exploratory Data Analysis (EDA)
==============================================================================
An end-to-end Machine Learning pipeline for predicting Rural Trip Generation.

This script performs a comprehensive exploratory analysis of the rural
household survey dataset (DISSR), including:

  1. Data loading, shape inspection, and summary statistics
  2. Missing-value analysis
  3. Distribution of every numerical variable (histograms with KDE)
  4. Full correlation heatmap with hierarchical clustering
  5. Pairwise scatter matrix of key socio-economic features
  6. Box plots of trip generation by categorical variables
  7. Violin plots of income across villages
  8. Target variable distribution analysis (skewness, kurtosis)
  9. Stratified train/test split with distribution verification
 10. All figures saved at 300 DPI for direct reporting inclusion

==============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import StratifiedShuffleSplit

import config as cfg
import utils

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING & INITIAL INSPECTION
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("1.  DATA LOADING & INITIAL INSPECTION")

df = pd.read_excel(cfg.DATA_FILE)

print(f"\n  Dataset Shape       : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Memory Usage        : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

utils.print_subsection("Column Data Types")
print(df.dtypes.to_string())

utils.print_subsection("First 5 Rows")
print(df.head().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 2.  DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("2.  DESCRIPTIVE STATISTICS")

desc_stats = df.describe().T
desc_stats["skew"] = df.select_dtypes(include=[np.number]).skew()
desc_stats["kurtosis"] = df.select_dtypes(include=[np.number]).kurtosis()

print(desc_stats.to_string())

# Export descriptive statistics as LaTeX table
utils.save_table_to_latex(
    desc_stats.round(2),
    filename="descriptive_statistics",
    caption="Descriptive Statistics of Survey Variables",
    label="tab:descriptive_stats",
    float_format="%.2f",
)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  MISSING VALUE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("3.  MISSING VALUE ANALYSIS")

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    "Missing Count": missing,
    "Missing (%)": missing_pct,
}).sort_values("Missing Count", ascending=False)

print(missing_df[missing_df["Missing Count"] > 0].to_string())
if missing.sum() == 0:
    print("  ✓ No missing values detected in the dataset.")

# Missing value heatmap
fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(df.isnull().T, cbar=True, cmap="YlOrRd",
            yticklabels=df.columns, ax=ax, cbar_kws={"shrink": 0.5})
ax.set_title("Missing Value Heatmap Across All Variables", fontsize=13, pad=15)
ax.set_xlabel("Sample Index", fontsize=11)
ax.set_ylabel("")
plt.tight_layout()
utils.save_figure("missing_value_heatmap", sub_dir="eda")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  NUMERICAL FEATURE DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("4.  NUMERICAL FEATURE DISTRIBUTIONS")

num_cols = df[cfg.NUMERICAL_FEATURES + [cfg.TARGET_COL]].select_dtypes(
    include=[np.number]
).columns.tolist()

n_cols_grid = 3
n_rows_grid = int(np.ceil(len(num_cols) / n_cols_grid))

fig, axes = plt.subplots(n_rows_grid, n_cols_grid,
                         figsize=(5 * n_cols_grid, 4 * n_rows_grid))
axes = axes.flatten()

colors = sns.color_palette("muted", len(num_cols))

for i, col in enumerate(num_cols):
    ax = axes[i]
    data = df[col].dropna()
    ax.hist(data, bins=25, color=colors[i], edgecolor="white",
            alpha=0.75, density=True, linewidth=0.5)
    # KDE overlay
    if data.nunique() > 2:
        try:
            kde_x = np.linspace(data.min(), data.max(), 200)
            kde = stats.gaussian_kde(data)
            ax.plot(kde_x, kde(kde_x), color="black", linewidth=1.5, alpha=0.8)
        except Exception:
            pass
    ax.set_title(col, fontsize=10, fontweight="bold")
    ax.set_ylabel("Density")
    ax.tick_params(labelsize=8)

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Distribution of Numerical Variables (with KDE Overlay)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
utils.save_figure("numerical_distributions", sub_dir="eda")

print(f"  ✓ Plotted distributions for {len(num_cols)} numerical features.")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("5.  CORRELATION HEATMAP")

numeric_df = df[cfg.NUMERICAL_FEATURES + [cfg.TARGET_COL]].select_dtypes(
    include=[np.number]
)
corr_matrix = numeric_df.corr()

# Mask the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(12, 10))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap=cmap,
    vmin=-1, vmax=1,
    center=0,
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=0.8,
    linecolor="white",
    cbar_kws={"shrink": 0.75, "label": "Pearson Correlation Coefficient"},
    ax=ax,
    annot_kws={"size": 9},
)
ax.set_title("Correlation Heatmap — Socio-Economic & Infrastructure Variables",
             fontsize=13, fontweight="bold", pad=20)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
utils.save_figure("correlation_heatmap", sub_dir="eda")

# LaTeX export of correlation matrix
utils.save_table_to_latex(
    corr_matrix.round(3),
    filename="correlation_matrix",
    caption="Pearson Correlation Matrix of Numerical Features",
    label="tab:correlation_matrix",
    float_format="%.3f",
)

# Print strongest correlations with the target
target_corr = corr_matrix[cfg.TARGET_COL].drop(cfg.TARGET_COL).abs().sort_values(
    ascending=False
)
utils.print_subsection("Strongest Correlations with Target (|r|)")
for feat, val in target_corr.items():
    utils.print_metric(feat, val)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  PAIRWISE SCATTER MATRIX — KEY FEATURES
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("6.  PAIRWISE SCATTER MATRIX")

scatter_features = [
    cfg.TARGET_COL,
    "Annual income(Rs)",
    "Population",
    "Persons employed in your household",
    "Persons involved in farming",
    "No of vehicles in household",
]

g = sns.pairplot(
    df[scatter_features].dropna(),
    diag_kind="kde",
    corner=True,
    plot_kws={"alpha": 0.4, "s": 20, "edgecolor": "none"},
    diag_kws={"fill": True, "alpha": 0.6},
    palette="muted",
)
g.fig.suptitle("Pairwise Scatter Matrix — Key Socio-Economic Features",
               fontsize=14, fontweight="bold", y=1.02)
utils.save_figure("pairwise_scatter_matrix", sub_dir="eda")
print("  ✓ Pairwise scatter matrix generated.")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  BOX PLOTS — TARGET BY CATEGORICAL VARIABLES
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("7.  BOX PLOTS — TRIPS BY CATEGORICAL VARIABLES")

cat_features_to_plot = ["Land use type", "Pavement condition",
                        "Name of Village", "Vehicle use to travel"]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, cat_col in enumerate(cat_features_to_plot):
    ax = axes[i]
    order = df.groupby(cat_col)[cfg.TARGET_COL].median().sort_values(
        ascending=False
    ).index
    sns.boxplot(
        data=df, x=cat_col, y=cfg.TARGET_COL, ax=ax,
        order=order, palette="Set2", linewidth=0.8,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
    )
    ax.set_title(f"Trip Generation by {cat_col}", fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Trips per Day (PCU)")
    ax.tick_params(axis="x", rotation=35, labelsize=8)

fig.suptitle("Distribution of Daily Trips (PCU) Across Categorical Variables",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
utils.save_figure("boxplots_trips_by_category", sub_dir="eda")

# 7a. STATISTICAL VALIDATION — ONE-WAY ANOVA
utils.print_subsection("One-Way ANOVA Test Results")
for cat_col in cat_features_to_plot:
    # Filter out small groups to avoid errors
    valid_groups = [
        df[df[cat_col] == val][cfg.TARGET_COL].dropna() 
        for val in df[cat_col].unique() 
        if len(df[df[cat_col] == val]) > 1
    ]
    if len(valid_groups) > 1:
        f_stat, p_val = stats.f_oneway(*valid_groups)
        print(f"  {cat_col:<25} : F={f_stat:.2f}, p={p_val:.4e}")
        if p_val < 0.05:
            print(f"    → Significant difference in trips between groups (p < 0.05)")
        else:
            print(f"    → No significant difference (p >= 0.05)")
    else:
        print(f"  {cat_col:<25} : Skipped (insufficient groups/data)")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  VIOLIN PLOTS — INCOME DISTRIBUTION ACROSS VILLAGES
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("8.  VIOLIN PLOTS — INCOME BY VILLAGE")

fig, ax = plt.subplots(figsize=(14, 6))
order = df.groupby("Name of Village")["Annual income(Rs)"].median().sort_values(
    ascending=False
).index
sns.violinplot(
    data=df, x="Name of Village", y="Annual income(Rs)", ax=ax,
    order=order, palette="coolwarm", inner="quartile", linewidth=0.7,
    cut=0,
)
ax.set_title("Annual Income Distribution Across Villages",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("")
ax.set_ylabel("Annual Income (₹)")
ax.tick_params(axis="x", rotation=35, labelsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
plt.tight_layout()
utils.save_figure("violin_income_by_village", sub_dir="eda")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  TARGET VARIABLE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("9.  TARGET VARIABLE ANALYSIS")

target = df[cfg.TARGET_COL].dropna()

print(f"  Mean                : {target.mean():.3f}")
print(f"  Median              : {target.median():.3f}")
print(f"  Std Dev             : {target.std():.3f}")
print(f"  Skewness            : {target.skew():.3f}")
print(f"  Kurtosis            : {target.kurtosis():.3f}")
print(f"  Min / Max           : {target.min():.1f} / {target.max():.1f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (a) Histogram + KDE
axes[0].hist(target, bins=25, color="#5C6BC0", edgecolor="white",
             alpha=0.75, density=True, linewidth=0.5)
kde_x = np.linspace(target.min(), target.max(), 200)
kde = stats.gaussian_kde(target)
axes[0].plot(kde_x, kde(kde_x), color="#EF5350", linewidth=2)
axes[0].set_title("(a) Distribution of Target Variable", fontweight="bold")
axes[0].set_xlabel("Trips per Day (PCU)")
axes[0].set_ylabel("Density")

# (b) Box plot
sns.boxplot(y=target, ax=axes[1], color="#66BB6A", linewidth=0.8,
            flierprops={"marker": "o", "markersize": 4})
axes[1].set_title("(b) Box Plot — Outlier Detection", fontweight="bold")
axes[1].set_ylabel("Trips per Day (PCU)")

# (c) Q-Q plot
stats.probplot(target, dist="norm", plot=axes[2])
axes[2].set_title("(c) Q-Q Plot — Normality Assessment", fontweight="bold")
axes[2].get_lines()[0].set(color="#42A5F5", markersize=4, alpha=0.6)
axes[2].get_lines()[1].set(color="#EF5350", linewidth=1.5)

fig.suptitle("Comprehensive Analysis of Target Variable (Trips_Per_Day_PCU)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
utils.save_figure("target_variable_analysis", sub_dir="eda")

# ─────────────────────────────────────────────────────────────────────────────
# 10.  STRATIFIED TRAIN/TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("10.  STRATIFIED TRAIN/TEST SPLIT")

# Create income categories for stratification
df["income_cat"] = pd.cut(
    df["Annual income(Rs)"],
    bins=cfg.INCOME_BINS,
    labels=cfg.INCOME_LABELS,
)

split = StratifiedShuffleSplit(
    n_splits=cfg.N_SPLITS,
    test_size=cfg.TEST_SIZE,
    random_state=cfg.RANDOM_STATE,
)

for train_idx, test_idx in split.split(df, df["income_cat"]):
    strat_train = df.loc[train_idx]
    strat_test = df.loc[test_idx]

print(f"  Training set size   : {len(strat_train)} ({len(strat_train)/len(df)*100:.1f}%)")
print(f"  Test set size       : {len(strat_test)} ({len(strat_test)/len(df)*100:.1f}%)")

# Verify stratification — compare income distributions
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
sets = [("Full Dataset", df), ("Training Set", strat_train), ("Test Set", strat_test)]
colors = ["#5C6BC0", "#66BB6A", "#FFA726"]

for i, (name, data) in enumerate(sets):
    props = data["income_cat"].value_counts(normalize=True).sort_index()
    axes[i].bar(props.index.astype(str), props.values, color=colors[i],
                edgecolor="white", linewidth=0.5, alpha=0.85)
    axes[i].set_title(f"{name} (n={len(data)})", fontweight="bold")
    axes[i].set_xlabel("Income Category")
    axes[i].set_ylabel("Proportion")
    axes[i].set_ylim(0, max(props.values) * 1.2)
    for j, (cat, val) in enumerate(props.items()):
        axes[i].text(j, val + 0.005, f"{val:.1%}", ha="center", fontsize=8)

fig.suptitle("Stratified Sampling Verification — Income Category Proportions",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
utils.save_figure("stratified_split_verification", sub_dir="eda")

# Clean up temporary column
for s in (strat_train, strat_test, df):
    s.drop("income_cat", axis=1, inplace=True, errors="ignore")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("EDA COMPLETE")
print("  All exploratory analysis figures have been saved to:")
print(f"    → {cfg.EDA_FIGURES_DIR}")
print(f"    → {cfg.LATEX_DIR}")
print()
