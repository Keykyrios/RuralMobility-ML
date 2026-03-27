"""
==============================================================================
Utility Functions — Rural Trip Generation Prediction Pipeline
==============================================================================
An end-to-end Machine Learning pipeline for predicting Rural Trip Generation.

Shared helper functions for figure saving, LaTeX table generation,
formatted console output, and directory management.
==============================================================================
"""

import os
import textwrap
import matplotlib.pyplot as plt
import config as cfg


# ─── Directory Management ───────────────────────────────────────────────────

def create_output_dirs():
    """
    Create all output directories defined in config.  Safe to call
    multiple times — existing directories are silently skipped.
    """
    dirs = [
        cfg.OUTPUT_DIR,
        cfg.FIGURES_DIR,
        cfg.RESULTS_DIR,
        cfg.LATEX_DIR,
        cfg.MODELS_DIR,
        cfg.EDA_FIGURES_DIR,
        cfg.MODEL_FIGURES_DIR,
        cfg.EVAL_FIGURES_DIR,
        cfg.PREDICTION_FIGURES_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ─── Figure Saving ──────────────────────────────────────────────────────────

def save_figure(fig_name, sub_dir=None, close=True):
    """
    Save the current Matplotlib figure as a high-resolution PNG.

    Parameters
    ----------
    fig_name : str
        File name (without extension).  Spaces are replaced with underscores.
    sub_dir : str, optional
        Subdirectory inside `output/figures/`.  If None, saves directly
        into `output/figures/`.
    close : bool
        Whether to close the figure after saving (default True).
    """
    create_output_dirs()
    safe_name = fig_name.replace(" ", "_").lower()
    target_dir = os.path.join(cfg.FIGURES_DIR, sub_dir) if sub_dir else cfg.FIGURES_DIR
    os.makedirs(target_dir, exist_ok=True)
    filepath = os.path.join(target_dir, f"{safe_name}.{cfg.FIGURE_FORMAT}")
    plt.savefig(filepath, dpi=cfg.FIGURE_DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"    ✓ Figure saved → {filepath}")
    if close:
        plt.close()


# ─── LaTeX Table Export ─────────────────────────────────────────────────────

def save_table_to_latex(df, filename, caption=None, label=None,
                        float_format="%.4f"):
    """
    Export a pandas DataFrame as a publication-ready LaTeX table using the
    ``booktabs`` package.  The resulting .tex file can be directly
    \\input{} into a thesis document.

    Parameters
    ----------
    df : pd.DataFrame
        The table data.
    filename : str
        Base file name (without extension).
    caption : str, optional
        LaTeX table caption.
    label : str, optional
        LaTeX label for cross-referencing (e.g. ``tab:model_metrics``).
    float_format : str
        printf-style format string for floats (default ``%.4f``).

    Returns
    -------
    str
        Absolute path of the saved .tex file.
    """
    create_output_dirs()
    safe_name = filename.replace(" ", "_").lower()
    filepath = os.path.join(cfg.LATEX_DIR, f"{safe_name}.tex")

    # Generate booktabs LaTeX via pandas
    latex_body = df.to_latex(
        index=True,
        float_format=float_format,
        caption=caption,
        label=label,
        bold_rows=True,
        column_format="l" + "r" * len(df.columns),
    )

    # Wrap in a standalone-friendly preamble comment
    header = textwrap.dedent(f"""\
    %% ──────────────────────────────────────────────────────────────────
    %% Auto-generated LaTeX table — Rural Trip Generation Prediction
    %% Copy this file into your thesis:  \\input{{{safe_name}}}
    %% Requires: \\usepackage{{booktabs}}
    %% ──────────────────────────────────────────────────────────────────
    """)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(latex_body)

    print(f"    ✓ LaTeX table saved → {filepath}")
    return filepath


# ─── Console Formatting ─────────────────────────────────────────────────────

def print_section_header(title, width=70):
    """Print a visually distinct section header to the console."""
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subsection(title, width=70):
    """Print a secondary-level header."""
    print()
    print(f"── {title} " + "─" * max(0, width - len(title) - 4))


def print_metric(name, value, fmt=".4f"):
    """Print a single metric line with consistent alignment."""
    print(f"    {name:<40s} : {value:{fmt}}")


def print_divider(width=70):
    """Print a thin divider line."""
    print("─" * width)


# ─── Quick Self-Test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd

    create_output_dirs()
    print_section_header("Utility Module Self-Test")

    # Test LaTeX export
    sample = pd.DataFrame({
        "Model":  ["Random Forest", "XGBoost"],
        "R²":     [0.8712, 0.8934],
        "RMSE":   [1.2345, 1.1023],
    }).set_index("Model")

    save_table_to_latex(
        sample,
        filename="sample_metrics",
        caption="Sample Model Comparison Metrics",
        label="tab:sample_metrics",
    )
    print("\n  All utilities working correctly.")
