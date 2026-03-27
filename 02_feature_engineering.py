"""
==============================================================================
Script 02 — Feature Engineering & Preprocessing Pipeline
==============================================================================
An end-to-end Machine Learning pipeline for predicting Rural Trip Generation.

This script constructs the complete data preprocessing pipeline:

  1. Reload data and recreate the stratified train/test split
  2. Engineer new composite features from domain knowledge
  3. Build numerical and categorical preprocessing pipelines
  4. Fit the full ColumnTransformer on training data
  5. Visualize feature distributions before and after scaling
  6. Export the prepared feature matrix

The pipeline object and feature names are saved for reuse in subsequent
scripts via pickle serialisation.
==============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

import config as cfg
import utils

# ─────────────────────────────────────────────────────────────────────────────
# 1.  RELOAD DATA & RECREATE STRATIFIED SPLIT
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("1.  LOADING DATA & STRATIFIED SPLIT")

def cap_outliers(data, key_features=None, lower_q=0.01, upper_q=0.99):
    """
    Cap extreme outliers in numerical features to the 1st and 99th percentiles
    to prevent skewing linear models.
    """
    df_capped = data.copy()
    if key_features is None:
        key_features = cfg.NUMERICAL_FEATURES + [cfg.TARGET_COL]
    
    utils.print_subsection("Outlier Capping (1st - 99th Percentile)")
    for col in key_features:
        if col in df_capped.columns:
            lower = df_capped[col].quantile(lower_q)
            upper = df_capped[col].quantile(upper_q)
            
            # Count outliers
            outliers_low = (df_capped[col] < lower).sum()
            outliers_high = (df_capped[col] > upper).sum()
            
            if outliers_low + outliers_high > 0:
                df_capped[col] = df_capped[col].clip(lower, upper)
                print(f"  {col:<35} : Capped {outliers_low + outliers_high} values ({lower:.1f} - {upper:.1f})")
    return df_capped

df = pd.read_excel(cfg.DATA_FILE)
df = cap_outliers(df)

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
    strat_train_set = df.loc[train_idx].copy()
    strat_test_set = df.loc[test_idx].copy()

# Drop income_cat helper column
for s in (strat_train_set, strat_test_set, df):
    s.drop("income_cat", axis=1, inplace=True, errors="ignore")

print(f"  Training samples: {len(strat_train_set)}")
print(f"  Test samples    : {len(strat_test_set)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE ENGINEERING — DOMAIN-DRIVEN COMPOSITE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("2.  FEATURE ENGINEERING")


def engineer_features(data):
    """
    Create domain-driven composite features from the raw survey data.

    New features:
      - household_size         : Males + Females (total household members)
      - income_per_capita      : Annual income / household_size
      - employment_ratio       : Employed persons / household_size
      - farming_intensity      : Farming persons / employed persons
      - accessibility_index    : Inverse weighted sum of distance to highway
                                 and distance to railway station
      - road_quality_indicator : Road width × pavement encoding
    """
    d = data.copy()

    # Household size
    d["household_size"] = (
        d["Males in your Household"].fillna(0)
        + d["Females in your household"].fillna(0)
    ).replace(0, 1)  # avoid division by zero

    # Income per capita
    d["income_per_capita"] = d["Annual income(Rs)"] / d["household_size"]

    # Employment ratio
    d["employment_ratio"] = (
        d["Persons employed in your household"] / d["household_size"]
    )

    # Farming intensity (persons in farming / employed persons)
    employed = d["Persons employed in your household"].replace(0, 1)
    d["farming_intensity"] = d["Persons involved in farming"] / employed

    # Accessibility index (lower distance → higher accessibility)
    hw_dist = d["Distance to nearest highway(Km)"].fillna(
        d["Distance to nearest highway(Km)"].median()
    )
    rw_dist = d["Distance to nearest Railway station(Km)"].fillna(
        d["Distance to nearest Railway station(Km)"].median()
    )
    d["accessibility_index"] = 1.0 / (1.0 + 0.6 * hw_dist + 0.4 * rw_dist)

    return d


strat_train_set = engineer_features(strat_train_set)
strat_test_set = engineer_features(strat_test_set)

engineered_features = [
    "household_size",
    "income_per_capita",
    "employment_ratio",
    "farming_intensity",
    "accessibility_index",
]

print("  New engineered features:")
for feat in engineered_features:
    train_vals = strat_train_set[feat].describe()
    print(f"    • {feat:<25s}  mean={train_vals['mean']:.3f}  "
          f"std={train_vals['std']:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  DEFINE FEATURE COLUMNS & PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("3.  PREPROCESSING PIPELINE")

# Full list of numerical features (original + engineered)
num_attribs = cfg.NUMERICAL_FEATURES + engineered_features
cat_attribs = cfg.CATEGORICAL_FEATURES

print(f"  Numerical features  : {len(num_attribs)}")
print(f"  Categorical features: {len(cat_attribs)}")

# Numerical pipeline: impute missing → scale
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("std_scaler", StandardScaler()),
])

# Full column transformer
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_attribs),
])

# ─────────────────────────────────────────────────────────────────────────────
# 4.  FIT & TRANSFORM
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("4.  FITTING PIPELINE ON TRAINING DATA")

X_train_raw = strat_train_set.drop(cfg.TARGET_COL, axis=1)
y_train = strat_train_set[cfg.TARGET_COL].copy()

X_test_raw = strat_test_set.drop(cfg.TARGET_COL, axis=1)
y_test = strat_test_set[cfg.TARGET_COL].copy()

X_train_prepared = full_pipeline.fit_transform(X_train_raw)
X_test_prepared = full_pipeline.transform(X_test_raw)

# Retrieve feature names after encoding
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_columns = list(cat_encoder.get_feature_names_out(cat_attribs))
all_feature_names = num_attribs + cat_columns

print(f"  Transformed X_train shape: {X_train_prepared.shape}")
print(f"  Transformed X_test shape : {X_test_prepared.shape}")
print(f"  Total feature count      : {len(all_feature_names)}")

# Convert to DataFrames for inspection
X_train_df = pd.DataFrame(X_train_prepared, columns=all_feature_names,
                           index=X_train_raw.index)
X_test_df = pd.DataFrame(X_test_prepared, columns=all_feature_names,
                          index=X_test_raw.index)

utils.print_subsection("Prepared Training Data — First 5 Rows (Numerical Only)")
print(X_train_df[num_attribs].head().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 5.  VISUALIZE: BEFORE vs AFTER SCALING
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("5.  FEATURE SCALING VISUALIZATION")

# Select 6 key numerical features for the comparison plot
vis_features = [
    "Annual income(Rs)",
    "Population",
    "Road width(m)",
    "income_per_capita",
    "employment_ratio",
    "accessibility_index",
]

fig, axes = plt.subplots(len(vis_features), 2, figsize=(14, 3 * len(vis_features)))

for i, feat in enumerate(vis_features):
    # Before scaling
    raw_data = X_train_raw[feat].dropna()
    axes[i, 0].hist(raw_data, bins=25, color="#42A5F5", edgecolor="white",
                     alpha=0.8, linewidth=0.5)
    axes[i, 0].set_title(f"{feat} — Raw", fontsize=9, fontweight="bold")
    axes[i, 0].set_ylabel("Count")

    # After scaling
    scaled_idx = all_feature_names.index(feat)
    scaled_data = X_train_prepared[:, scaled_idx]
    axes[i, 1].hist(scaled_data, bins=25, color="#66BB6A", edgecolor="white",
                     alpha=0.8, linewidth=0.5)
    axes[i, 1].set_title(f"{feat} — Scaled (μ=0, σ=1)", fontsize=9,
                          fontweight="bold")
    axes[i, 1].set_ylabel("Count")

fig.suptitle("Feature Distributions Before vs After StandardScaler",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
utils.save_figure("feature_scaling_comparison", sub_dir="eda")
print("  ✓ Before/after scaling visualization saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  ENGINEERED FEATURE CORRELATION WITH TARGET
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("6.  ENGINEERED FEATURES — CORRELATION WITH TARGET")

eng_corr_cols = engineered_features + [cfg.TARGET_COL]
eng_corr = strat_train_set[eng_corr_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(eng_corr, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
            linewidths=0.8, linecolor="white", ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Correlation: Engineered Features vs Target Variable",
             fontsize=12, fontweight="bold", pad=15)
plt.tight_layout()
utils.save_figure("engineered_features_correlation", sub_dir="eda")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  SAVE ARTEFACTS FOR DOWNSTREAM SCRIPTS
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("7.  SAVING PIPELINE ARTEFACTS")
utils.create_output_dirs()

artefacts = {
    "full_pipeline":      full_pipeline,
    "X_train":            X_train_prepared,
    "X_test":             X_test_prepared,
    "y_train":            y_train,
    "y_test":             y_test,
    "feature_names":      all_feature_names,
    "num_attribs":        num_attribs,
    "cat_attribs":        cat_attribs,
    "strat_train_set":    strat_train_set,
    "strat_test_set":     strat_test_set,
}

artefact_path = joblib.dump(artefacts,
                            f"{cfg.MODELS_DIR}/pipeline_artefacts.joblib")[0]
print(f"  ✓ Pipeline artefacts saved → {artefact_path}")

# Feature name table for LaTeX
feat_table = pd.DataFrame({
    "Feature": all_feature_names,
    "Type": ["Numerical"] * len(num_attribs) + ["Categorical (OHE)"] * len(cat_columns),
})
feat_table.index = range(1, len(feat_table) + 1)
feat_table.index.name = "No."

utils.save_table_to_latex(
    feat_table,
    filename="feature_list",
    caption="Complete List of Features Used in the Model",
    label="tab:feature_list",
)

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("FEATURE ENGINEERING COMPLETE")
print(f"  Original features   : {len(cfg.NUMERICAL_FEATURES) + len(cfg.CATEGORICAL_FEATURES)}")
print(f"  Engineered features : {len(engineered_features)}")
print(f"  Total after encoding: {len(all_feature_names)}")
print(f"  Pipeline saved to   : {cfg.MODELS_DIR}/")
print()
