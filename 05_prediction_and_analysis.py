"""
==============================================================================
Script 05 — Predictions & Policy-Level Analysis
==============================================================================
An end-to-end Machine Learning pipeline for predicting Rural Trip Generation.

This script produces the final outputs for the project:

  1. Household-level predictions for the full dataset
  2. Village-level aggregation  (total trips, average, household count)
  3. Village trip generation bar chart
  4. Road capacity analysis  (Road Width vs ADT)
  5. Income vs Trip Generation with regression line
  6. Land use type — grouped trip analysis
  7. Export prediction CSVs and LaTeX tables

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

from sklearn.model_selection import StratifiedShuffleSplit

import config as cfg
import utils

# Re-import engineer_features from Script 02
import importlib.util
spec = importlib.util.spec_from_file_location("fe", "02_feature_engineering.py")
# Instead of importing, we redefine the function here for standalone execution


def engineer_features(data):
    """Recreate the same engineered features as in Script 02."""
    d = data.copy()
    d["household_size"] = (
        d["Males in your Household"].fillna(0)
        + d["Females in your household"].fillna(0)
    ).replace(0, 1)
    d["income_per_capita"] = d["Annual income(Rs)"] / d["household_size"]
    d["employment_ratio"] = (
        d["Persons employed in your household"] / d["household_size"]
    )
    employed = d["Persons employed in your household"].replace(0, 1)
    d["farming_intensity"] = d["Persons involved in farming"] / employed
    hw_dist = d["Distance to nearest highway(Km)"].fillna(
        d["Distance to nearest highway(Km)"].median()
    )
    rw_dist = d["Distance to nearest Railway station(Km)"].fillna(
        d["Distance to nearest Railway station(Km)"].median()
    )
    d["accessibility_index"] = 1.0 / (1.0 + 0.6 * hw_dist + 0.4 * rw_dist)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# 0.  LOAD DATA & ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("0.  LOADING DATA & ARTEFACTS")

df = pd.read_excel(cfg.DATA_FILE)
pipe_art = joblib.load(f"{cfg.MODELS_DIR}/pipeline_artefacts.joblib")
model_art = joblib.load(f"{cfg.MODELS_DIR}/trained_models.joblib")

full_pipeline = pipe_art["full_pipeline"]
best_model_name = model_art["best_model_name"]
best_model = model_art["tuned_models"][best_model_name]

print(f"  Dataset size  : {len(df)} households")
print(f"  Best model    : {best_model_name}")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  FULL-DATASET PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("1.  HOUSEHOLD-LEVEL PREDICTIONS")

# Apply feature engineering to full dataset
df_eng = engineer_features(df)

X_full = df_eng.drop(cfg.TARGET_COL, axis=1)
X_full_prepared = full_pipeline.transform(X_full)

df["Predicted_Trips_PCU"] = best_model.predict(X_full_prepared)
df["Prediction_Error"] = df[cfg.TARGET_COL] - df["Predicted_Trips_PCU"]
df["Abs_Error"] = np.abs(df["Prediction_Error"])

print(f"  Predictions generated for {len(df)} households.")
print(f"  Mean Predicted Trips  : {df['Predicted_Trips_PCU'].mean():.3f}")
print(f"  Mean Actual Trips     : {df[cfg.TARGET_COL].mean():.3f}")
print(f"  Mean Absolute Error   : {df['Abs_Error'].mean():.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  VILLAGE-LEVEL AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("2.  VILLAGE-LEVEL AGGREGATION")

village_summary = df.groupby("Name of Village").agg(
    Total_Predicted_Trips=("Predicted_Trips_PCU", "sum"),
    Total_Actual_Trips=(cfg.TARGET_COL, "sum"),
    Avg_Trips_Per_Household=("Predicted_Trips_PCU", "mean"),
    Household_Count=("Predicted_Trips_PCU", "count"),
    Avg_Income=("Annual income(Rs)", "mean"),
    Avg_Road_Width=("Road width(m)", "mean"),
    Avg_Population=("Population", "mean"),
).round(3)

village_summary = village_summary.sort_values("Total_Predicted_Trips",
                                               ascending=False)

print(village_summary.to_string())

# LaTeX export
utils.save_table_to_latex(
    village_summary,
    filename="village_level_summary",
    caption="Village-Level Trip Generation Summary — Predicted vs Actual",
    label="tab:village_summary",
    float_format="%.2f",
)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  VILLAGE TRIP GENERATION — BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("3.  VILLAGE TRIP GENERATION BAR CHART")

fig, ax = plt.subplots(figsize=(12, 7))

vs = village_summary.sort_values("Total_Predicted_Trips")
colors = sns.color_palette("YlOrRd", len(vs))

ax.barh(
    range(len(vs)),
    vs["Total_Predicted_Trips"],
    color=colors,
    edgecolor="white",
    linewidth=0.5,
    height=0.65,
)
ax.set_yticks(range(len(vs)))
ax.set_yticklabels(vs.index, fontsize=10)
ax.set_xlabel("Total Predicted Trips (PCU/Day)", fontsize=11)
ax.set_title("Village-Level Trip Generation — Total Predicted ADT (PCU)",
             fontsize=13, fontweight="bold", pad=15)

# Annotate bar values
for i, (_, row) in enumerate(vs.iterrows()):
    ax.text(row["Total_Predicted_Trips"] + 0.2, i,
            f"{row['Total_Predicted_Trips']:.1f}  ({row['Household_Count']} HH)",
            va="center", fontsize=9, color="#333")

plt.tight_layout()
utils.save_figure("village_trip_generation", sub_dir="prediction")

# Side-by-side actual vs predicted
fig, ax = plt.subplots(figsize=(12, 7))
width = 0.35
x_pos = np.arange(len(vs))

ax.barh(x_pos - width / 2, vs["Total_Actual_Trips"], width,
        color="#42A5F5", edgecolor="white", label="Actual", alpha=0.85)
ax.barh(x_pos + width / 2, vs["Total_Predicted_Trips"], width,
        color="#EF5350", edgecolor="white", label="Predicted", alpha=0.85)
ax.set_yticks(x_pos)
ax.set_yticklabels(vs.index, fontsize=10)
ax.set_xlabel("Total Trips (PCU/Day)", fontsize=11)
ax.set_title("Village-Level Comparison — Actual vs Predicted Trip Generation",
             fontsize=13, fontweight="bold", pad=15)
ax.legend(fontsize=10)
plt.tight_layout()
utils.save_figure("village_actual_vs_predicted", sub_dir="prediction")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  ROAD CAPACITY ANALYSIS — ROAD WIDTH vs ADT
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("4.  ROAD CAPACITY ANALYSIS")

fig, ax = plt.subplots(figsize=(10, 7))

sns.scatterplot(
    data=village_summary.reset_index(),
    x="Avg_Road_Width", y="Total_Predicted_Trips",
    size="Household_Count", hue="Name of Village",
    sizes=(80, 500), alpha=0.75, ax=ax,
    palette="Set2", edgecolor="white", linewidth=0.5,
)

# Add village labels
for idx, row in village_summary.iterrows():
    ax.annotate(
        idx, (row["Avg_Road_Width"], row["Total_Predicted_Trips"]),
        fontsize=8, ha="center", va="bottom",
        xytext=(0, 8), textcoords="offset points",
    )

ax.set_xlabel("Average Road Width (m)", fontsize=11)
ax.set_ylabel("Total Predicted ADT (PCU)", fontsize=11)
ax.set_title("Road Capacity Analysis — Traffic Volume vs Road Infrastructure",
             fontsize=13, fontweight="bold", pad=15)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8,
          title="Village", title_fontsize=9)
plt.tight_layout()
utils.save_figure("road_capacity_analysis", sub_dir="prediction")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  INCOME vs TRIP GENERATION
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("5.  INCOME vs TRIP GENERATION")

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(df["Annual income(Rs)"], df["Predicted_Trips_PCU"],
           alpha=0.4, s=30, color="#5C6BC0", edgecolors="white",
           linewidth=0.3, zorder=3)

# Regression line with confidence band
x_vals = df["Annual income(Rs)"].values
y_vals = df["Predicted_Trips_PCU"].values
slope, intercept, r, p, se = stats.linregress(x_vals, y_vals)
x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
y_line = slope * x_line + intercept
ax.plot(x_line, y_line, color="#EF5350", linewidth=2,
        label=f"Regression (r={r:.3f}, p={p:.4f})")

# Confidence band
y_err = se * np.sqrt(1 / len(x_vals) + (x_line - x_vals.mean()) ** 2 / np.sum(
    (x_vals - x_vals.mean()) ** 2))
ax.fill_between(x_line, y_line - 1.96 * y_err, y_line + 1.96 * y_err,
                alpha=0.15, color="#EF5350")

ax.set_xlabel("Annual Household Income (₹)", fontsize=11)
ax.set_ylabel("Predicted Trips per Day (PCU)", fontsize=11)
ax.set_title("Relationship Between Income and Trip Generation",
             fontsize=13, fontweight="bold", pad=15)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
ax.legend(fontsize=10)
plt.tight_layout()
utils.save_figure("income_vs_trips", sub_dir="prediction")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  LAND USE TYPE — TRIP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("6.  LAND USE TYPE — TRIP ANALYSIS")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) Average trips by land use
lu_avg = df.groupby("Land use type")["Predicted_Trips_PCU"].mean().sort_values(
    ascending=False
)
colors_lu = sns.color_palette("viridis", len(lu_avg))
axes[0].barh(range(len(lu_avg)), lu_avg.values, color=colors_lu,
             edgecolor="white", linewidth=0.5)
axes[0].set_yticks(range(len(lu_avg)))
axes[0].set_yticklabels(lu_avg.index, fontsize=9)
axes[0].set_xlabel("Avg Predicted Trips (PCU/Day)")
axes[0].set_title("(a) Mean Trip Generation by Land Use Type", fontweight="bold")
for i, v in enumerate(lu_avg.values):
    axes[0].text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9)

# (b) Total trips by land use
lu_total = df.groupby("Land use type")["Predicted_Trips_PCU"].sum().sort_values(
    ascending=False
)
axes[1].barh(range(len(lu_total)), lu_total.values,
             color=sns.color_palette("magma", len(lu_total)),
             edgecolor="white", linewidth=0.5)
axes[1].set_yticks(range(len(lu_total)))
axes[1].set_yticklabels(lu_total.index, fontsize=9)
axes[1].set_xlabel("Total Predicted Trips (PCU/Day)")
axes[1].set_title("(b) Total Trip Generation by Land Use Type", fontweight="bold")
for i, v in enumerate(lu_total.values):
    axes[1].text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=9)

fig.suptitle("Trip Generation Analysis by Land Use Classification",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
utils.save_figure("land_use_trip_analysis", sub_dir="prediction")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  PAVEMENT CONDITION IMPACT
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("7.  PAVEMENT CONDITION IMPACT ON TRIPS")

fig, ax = plt.subplots(figsize=(10, 6))

pav_order = df.groupby("Pavement condition")["Predicted_Trips_PCU"].median(
).sort_values(ascending=False).index

sns.boxplot(
    data=df, x="Pavement condition", y="Predicted_Trips_PCU",
    order=pav_order, palette="RdYlGn", ax=ax, linewidth=0.8,
    flierprops={"marker": "D", "markersize": 3, "alpha": 0.4},
)
sns.stripplot(
    data=df, x="Pavement condition", y="Predicted_Trips_PCU",
    order=pav_order, color="#333", alpha=0.25, size=3, ax=ax, jitter=True,
)
ax.set_xlabel("Pavement Condition", fontsize=11)
ax.set_ylabel("Predicted Trips per Day (PCU)", fontsize=11)
ax.set_title("Impact of Pavement Condition on Trip Generation",
             fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
utils.save_figure("pavement_condition_impact", sub_dir="prediction")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  EXPORT RESULTS
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("8.  EXPORTING RESULTS")
utils.create_output_dirs()

# Household-level predictions
household_cols = [
    "Name of Village", "Population", "Males in your Household",
    "Females in your household", "Persons employed in your household",
    "Annual income(Rs)", "Land use type", "Vehicle use to travel",
    "Road width(m)", "Pavement condition",
    cfg.TARGET_COL, "Predicted_Trips_PCU", "Prediction_Error", "Abs_Error",
]
household_out = df[[c for c in household_cols if c in df.columns]]
household_path = f"{cfg.RESULTS_DIR}/household_predictions.csv"
household_out.to_csv(household_path, index=False)
household_out.to_excel(household_path.replace(".csv", ".xlsx"), index=False)
print(f"  ✓ Household predictions → {household_path} (and .xlsx)")

# Village summary
village_path = f"{cfg.RESULTS_DIR}/village_summary.csv"
village_summary.to_csv(village_path)
village_summary.to_excel(village_path.replace(".csv", ".xlsx"))
print(f"  ✓ Village summary      → {village_path} (and .xlsx)")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

utils.print_section_header("PREDICTION & ANALYSIS COMPLETE")
print(f"  Model Used         : {best_model_name}")
print(f"  Households         : {len(df)}")
print(f"  Villages           : {len(village_summary)}")
print(f"  Figures saved to   : {cfg.PREDICTION_FIGURES_DIR}")
print(f"  CSVs saved to      : {cfg.RESULTS_DIR}")
print(f"  LaTeX tables       : {cfg.LATEX_DIR}")
print()
print("  ══════════════════════════════════════════════════════════════")
print("  ✓  ALL 5 SCRIPTS COMPLETE — ML PIPELINE FINISHED")
print("  ══════════════════════════════════════════════════════════════")
print()
