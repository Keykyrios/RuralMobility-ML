"""
==============================================================================
Configuration Module — Rural Trip Generation Prediction Pipeline
==============================================================================
An end-to-end Machine Learning pipeline for predicting Rural Trip Generation.

Centralized configuration for file paths, feature definitions, model
hyperparameters, and publication-quality plot styling.
==============================================================================
"""

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ─── Random Seed (Reproducibility) ──────────────────────────────────────────
RANDOM_STATE = 42

# ─── File Paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "DISSR.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
LATEX_DIR = os.path.join(OUTPUT_DIR, "latex_tables")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

# Sub-directories for organized figure storage
EDA_FIGURES_DIR = os.path.join(FIGURES_DIR, "eda")
MODEL_FIGURES_DIR = os.path.join(FIGURES_DIR, "model")
EVAL_FIGURES_DIR = os.path.join(FIGURES_DIR, "evaluation")
PREDICTION_FIGURES_DIR = os.path.join(FIGURES_DIR, "prediction")

# ─── Target Variable ────────────────────────────────────────────────────────
TARGET_COL = "Trips_Per_Day_PCU"

# ─── Feature Definitions ────────────────────────────────────────────────────
NUMERICAL_FEATURES = [
    "Population",
    "Males in your Household",
    "Females in your household",
    "Persons employed in your household",
    "Annual income(Rs)",
    "Persons involved in farming",
    "No of vehicles in household",
    "Distance to nearest highway(Km)",
    "Distance to nearest Railway station(Km)",
    "Road width(m)",
]

CATEGORICAL_FEATURES = [
    "Name of Village",
    "Land use type",
    "Vehicle use to travel",
    "Transportation of crops you grow",
    "Pavement condition",
]

# Income bins for stratified sampling
INCOME_BINS = [0, 150000, 300000, 450000, 600000, 750000, float("inf")]
INCOME_LABELS = [1, 2, 3, 4, 5, 6]

# ─── Train/Test Split ───────────────────────────────────────────────────────
TEST_SIZE = 0.20
N_SPLITS = 1

# ─── Cross-Validation ───────────────────────────────────────────────────────
CV_FOLDS = 10

# ─── Publication-Quality Plot Styling ────────────────────────────────────────
FIGURE_DPI = 300
FIGURE_FORMAT = "png"

# Professional color palette for models
MODEL_COLORS = {
    "Dummy (Baseline)":     "#B0BEC5",
    "Linear Regression":    "#42A5F5",
    "Ridge Regression":     "#5C6BC0",
    "Lasso Regression":     "#7E57C2",
    "Decision Tree":        "#66BB6A",
    "Random Forest":        "#26A69A",
    "Gradient Boosting":    "#FFA726",
    "XGBoost":              "#EF5350",
    "SVR":                  "#EC407A",
}

# Seaborn / Matplotlib global theme
PALETTE = "muted"
CONTEXT = "paper"


def apply_plot_style():
    """
    Apply a publication-quality Matplotlib/Seaborn theme suitable for
    technical reports and presentations. Uses serif fonts, muted colors,
    and clean grids.
    """
    sns.set_theme(
        style="whitegrid",
        context=CONTEXT,
        palette=PALETTE,
        font="serif",
    )

    plt.rcParams.update({
        # ── Font ──
        "font.family":          "serif",
        "font.size":            11,
        "axes.titlesize":       13,
        "axes.labelsize":       11,
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        "legend.fontsize":      9,

        # ── Figure ──
        "figure.dpi":           FIGURE_DPI,
        "savefig.dpi":          FIGURE_DPI,
        "savefig.bbox":         "tight",
        "figure.figsize":       (10, 6),

        # ── Grid ──
        "axes.grid":            True,
        "grid.linestyle":       "--",
        "grid.alpha":           0.35,
        "grid.linewidth":       0.5,

        # ── Axes ──
        "axes.edgecolor":       "#333333",
        "axes.linewidth":       0.8,
        "axes.spines.top":      False,
        "axes.spines.right":    False,

        # ── Legend ──
        "legend.framealpha":    0.9,
        "legend.edgecolor":     "#cccccc",
    })


# Apply style on import so every script inherits it
apply_plot_style()


if __name__ == "__main__":
    print("=" * 70)
    print("  Configuration Module — Rural Trip Generation Prediction")
    print("=" * 70)
    print(f"  Data File   : {DATA_FILE}")
    print(f"  Output Dir  : {OUTPUT_DIR}")
    print(f"  Random Seed : {RANDOM_STATE}")
    print(f"  Test Size   : {TEST_SIZE}")
    print(f"  CV Folds    : {CV_FOLDS}")
    print(f"  Figure DPI  : {FIGURE_DPI}")
    print(f"  Num Features: {len(NUMERICAL_FEATURES)}")
    print(f"  Cat Features: {len(CATEGORICAL_FEATURES)}")
    print("=" * 70)
