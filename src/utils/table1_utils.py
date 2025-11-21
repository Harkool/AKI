"""
Utility functions for generating baseline characteristics tables (Table 1)
Separate module for clean import in notebooks.
"""

import pandas as pd
import numpy as np
from scipy import stats

__all__ = ["generate_table1", "generate_table1_stage"]

# ------------------------------------------------------------
# P‑value helper
# ------------------------------------------------------------
def _compute_p_value(df, col, group_col, numeric_cols, binary_cols):
    groups = sorted(df[group_col].unique())
    if len(groups) != 2:
        raise ValueError("group_col must have exactly 2 groups for this test.")

    g1, g2 = groups
    data1 = df[df[group_col] == g1][col].dropna()
    data2 = df[df[group_col] == g2][col].dropna()

    # Continuous variables
    if col in numeric_cols:
        try:
            stat, p = stats.ttest_ind(data1, data2, equal_var=False)
        except Exception:
            p = np.nan

    # Binary / categorical
    elif col in binary_cols:
        contingency = pd.crosstab(df[col], df[group_col])
        if contingency.shape == (2, 2):
            _, p = stats.fisher_exact(contingency)
        else:
            _, p, _, _ = stats.chi2_contingency(contingency)

    else:
        p = np.nan

    return p

# ------------------------------------------------------------
# Table 1 (binary group)
# ------------------------------------------------------------
def generate_table1(df, group_col="aki_bin", p_value_threshold=0.05):
    """
    Generates a complete Table 1 comparing two groups (e.g., AKI vs No AKI).

    Returns:
        pd.DataFrame
    """

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["id", "patient_id", group_col]
    features = [c for c in numeric_cols if c not in exclude_cols]

    # Identify binary categorical
    binary_cols = [c for c in features if df[c].nunique() == 2 and set(df[c].unique()) <= {0, 1}]

    # Continuous vars
    continuous_cols = [c for c in features if c not in binary_cols]

    grouped = df.groupby(group_col)
    groups = sorted(df[group_col].unique())

    table_rows = []
    total_n = len(df)
    n0 = len(grouped.get_group(groups[0]))
    n1 = len(grouped.get_group(groups[1]))

    # Header row
    table_rows.append(pd.Series({
        "Feature": "Total Patients",
        f"Overall (N={total_n})": "",
        f"{group_col}={groups[0]} (n={n0})": "",
        f"{group_col}={groups[1]} (n={n1})": "",
        "p-value": ""
    }))

    # Continuous variables
    for col in continuous_cols:
        mean_sd_all = f"{df[col].mean():.2f} ± {df[col].std():.2f}"
        mean_sd_0 = f"{grouped.get_group(groups[0])[col].mean():.2f} ± {grouped.get_group(groups[0])[col].std():.2f}"
        mean_sd_1 = f"{grouped.get_group(groups[1])[col].mean():.2f} ± {grouped.get_group(groups[1])[col].std():.2f}"

        p = _compute_p_value(df, col, group_col, continuous_cols, binary_cols)
        p_str = f"{p:.3f}" if p >= 0.001 else "<0.001"

        table_rows.append(pd.Series({
            "Feature": col,
            f"Overall (N={total_n})": mean_sd_all,
            f"{group_col}={groups[0]} (n={n0})": mean_sd_0,
            f"{group_col}={groups[1]} (n={n1})": mean_sd_1,
            "p-value": p_str,
            "_p_num": p
        }))

    # Binary categorical
    for col in binary_cols:
        total_count = df[col].sum()
        pct_total = total_count / total_n * 100
        val_all = f"{total_count} ({pct_total:.1f}%)"

        c0 = grouped.get_group(groups[0])[col].sum()
        p0 = c0 / n0 * 100
        val_0 = f"{c0} ({p0:.1f}%)"

        c1 = grouped.get_group(groups[1])[col].sum()
        p1 = c1 / n1 * 100
        val_1 = f"{c1} ({p1:.1f}%)"

        p = _compute_p_value(df, col, group_col, continuous_cols, binary_cols)
        p_str = f"{p:.3f}" if p >= 0.001 else "<0.001"

        table_rows.append(pd.Series({
            "Feature": col,
            f"Overall (N={total_n})": val_all,
            f"{group_col}={groups[0]} (n={n0})": val_0,
            f"{group_col}={groups[1]} (n={n1})": val_1,
            "p-value": p_str,
            "_p_num": p
        }))

    table = pd.DataFrame(table_rows).set_index("Feature")

    # Bold significant p-values
    def _bold(series):
        p = series.get("_p_num", np.nan)
        s = series["p-value"]
        if not pd.isna(p) and p < p_value_threshold:
            return f"**{s}**"
        return s

    table["p-value"] = table.apply(_bold, axis=1)
    table = table.drop(columns=["_p_num"], errors="ignore")

    return table

# ------------------------------------------------------------
# Table 1 (multi‑group: AKI stage 0/1/2/3)
# ------------------------------------------------------------
def generate_table1_stage(df, group_col="aki_stage"):
    groups = sorted(df[group_col].unique())
    table_rows = []
    total_n = len(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["id", "patient_id", group_col]]

    # Header
    header = {
        "Feature": "Total Patients",
        f"Overall (N={total_n})": ""
    }
    for g in groups:
        header[f"{group_col}={g} (n={len(df[df[group_col]==g])})"] = ""
    header["p-value"] = ""
    table_rows.append(pd.Series(header))

    # Variables
    for col in numeric_cols:
        row = {"Feature": col}

        # Continuous stats
        row[f"Overall (N={total_n})"] = f"{df[col].mean():.2f} ± {df[col].std():.2f}"

        for g in groups:
            vals = df[df[group_col] == g][col]
            row[f"{group_col}={g} (n={len(vals)})"] = f"{vals.mean():.2f} ± {vals.std():.2f}"

        # ANOVA
        group_vals = [df[df[group_col] == g][col].dropna() for g in groups]
        try:
            _, p = stats.f_oneway(*group_vals)
        except Exception:
            p = np.nan

        row["p-value"] = f"{p:.3f}" if p >= 0.001 else "<0.001"

        table_rows.append(pd.Series(row))

    return pd.DataFrame(table_rows).set_index("Feature")
