#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.metrics import r2_score

plt.ion()
sns.set_style("whitegrid")
colors = sns.color_palette("colorblind")
markers = ["o", "x", "^", "+", "*", "p", "D", "V", "8"]

np.set_printoptions(linewidth=150)
pd.options.display.max_rows = 300
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_info_columns", 500)
pd.set_option("styler.render.max_columns", 500)
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 165)


def plot_rr_by_condition(
    metric_q,
    age_bin,
    ratedf,  # ratedf_all, ratedf_vo2havers, ratedf_vo2...
    sex_to_plot="Male",
    condition_order=None,
    ax=None,
):
    """Make tall forest plot, comparing RR over all conditions available"""
    if ax is None:
        _, ax = plt.figure(figsize=(4, 12)), plt.gca()

    ## sort conditions roughly by average RR
    if condition_order is None:
        condition_order = (
            ratedf[(ratedf.bidx == -1) & (ratedf.metric_q == 0)]  #  type: ignore
            .groupby("condition")
            .agg(rr_ave=("rel_risk", "mean"))
            .sort_values(by="rr_ave", ascending=False)
            .index.tolist()
        )

    # select (condition, sex) and remove oldest age group
    pdf = ratedf.reset_index()
    pdf = pdf[
        (pdf.health_metric == "adj_resid")
        & (pdf.metric_q.isin([0, 4]))
        & (pdf.sex == sex_to_plot)
    ]

    # remove low and high age groups
    alo, ahi = pdf.age_bin.cat.categories[0], pdf.age_bin.cat.categories[-1]
    pdf = pdf[~pdf.age_bin.isin([alo, ahi]) & (pdf.age_bin == age_bin)].assign(
        age_bin=lambda x: x.age_bin.cat.remove_unused_categories(),
        condition=lambda x: pd.Categorical(x.condition, categories=condition_order),
    )

    # plot metric stratification by
    sns.pointplot(
        x="rel_risk",
        y="condition",
        hue="metric_q",
        data=pdf,
        errorbar=("pi", 95),
        join=False,  # type: ignore
        dodge=0.4,  # type: ignore
        ax=ax,
        palette=sns.color_palette("rocket", 2),
        scale=0.8,  # type: ignore
        markers=markers,  # type: ignore
    )
    ylo, yhi = ax.get_ylim()  # type: ignore
    ax.vlines(x=1.0, ymin=ylo, ymax=yhi, color="grey", linestyle="--")  # type: ignore
    ax.set_ylim(ylo, yhi)  # type: ignore
    ax.set_title(  # type: ignore
        f"Age Gap Rel. Risk \n(chrono age = {age_bin}, sex={sex_to_plot})"
    )  #  type: ignore
    ax.set_xlabel("Rel. Risk")  # type: ignore
    ax.set_ylabel("Condition (self reported)")  # type: ignore
    ax.legend(title="Age Gap Quint.")  # type: ignore


def plot_stat_by_age_group(
    condition,  # in medhist_cols,
    sex_to_plot,  # Male/Female
    metric_to_plot,  # adj_resid, vo2max_q,
    stat_to_plot,  # rate | rel_risk | abs_risk_inc | rel_risk_inc,
    ratedf,  # ratedf_all, ratedf_vo2havers, ratedf_vo2...
    x_var="age_bin",
    ax=None,
):
    """Figure 2 plot --- comparison statistics for subgroups, comparing
    to mean baseline (e.g., rate of diabetes, rel risk of diabetes, abs risk inc, etc)

    Args
        - condition to isolate (e.g., medhist col)
        - sex_to_plot: "Male" or "Female"
        - metric_to_plot: "adj_resid" (age gap) or "vo2max_q" (vo2max quintile)
        - stat_to_plot: "rate|rel_risk|abs_risk_inc|rel_risk_inc
        - ratedf: dataframe output of relrisk.py, bootstrapped risks
        - x_var: "age_bin" or "vo2max_q" for different plots
    """
    if ax is None:
        _, ax = plt.figure(figsize=(10, 4)), plt.gca()

    # select (condition, sex) and remove oldest age group
    pdf = ratedf.reset_index()
    pdf = pdf[
        (pdf.condition == condition)
        & (pdf.health_metric == metric_to_plot)
        & (pdf.sex == sex_to_plot)
    ]

    # remove low and high age groups
    alo, ahi = pdf.age_bin.cat.categories[0], pdf.age_bin.cat.categories[-1]
    pdf = pdf[~pdf.age_bin.isin([alo, ahi])].assign(
        age_bin=lambda x: x.age_bin.cat.remove_unused_categories()
    )

    # plot metric stratification by
    sns.pointplot(
        x=x_var,
        y=stat_to_plot,
        hue="metric_q",
        data=pdf,
        errorbar=("pi", 95),
        join=False,  # type: ignore
        dodge=0.6,  # type: ignore
        ax=ax,
        palette=sns.color_palette("rocket", 5),
        markers=markers,  # type: ignore
    )

    if stat_to_plot == "rate":
        sns.pointplot(
            x=x_var,
            y="base_rate",
            data=pdf[pdf.bidx == -1],
            ax=ax,
            color="grey",
            linestyles="--",  # type: ignore
        )
        ax.set_ylabel("Rate")  # type: ignore
        ax.set_title(f"{condition}: diagnosis rate (sex={sex_to_plot})")  # type: ignore

    if stat_to_plot == "rel_risk":
        xlo, xhi = ax.get_xlim()  # type: ignore
        ax.hlines(1.0, xlo, xhi, linestyles="--", color="grey")  # type: ignore
        ax.set_xlim(xlo, xhi)  # type: ignore
        ax.set_ylabel("Rel. Risk")  # type: ignore
        ax.set_title(f"{condition}: relative risk (sex={sex_to_plot})")  # type: ignore

    if stat_to_plot == "abs_risk_inc":
        xlo, xhi = ax.get_xlim()  # type: ignore
        ax.hlines(0.0, xlo, xhi, linestyles="--", color="grey")  # type: ignore
        ax.set_xlim(xlo, xhi)  # type: ignore
        ax.set_ylabel("Abs. Risk Inc")  # type: ignore
        ax.set_title(  # type: ignore
            f"{condition}: absolute risk increase (sex={sex_to_plot})"
        )  # type:  ignore

    if stat_to_plot == "rel_risk_inc":
        xlo, xhi = ax.get_xlim()  # type: ignore
        ax.hlines(0.0, xlo, xhi, linestyles="--", color="grey")  # type: ignore
        ax.set_xlim(xlo, xhi)  # type: ignore
        ax.set_ylabel("Rel. Risk Inc")  # type: ignore
        ax.set_title(f"{condition}: relative risk increase (sex={sex_to_plot})")  # type: ignore

    ax.legend(title="Age Gap Quint.", bbox_to_anchor=(0.97, 0.7))  # type: ignore
    if x_var == "age_bin":
        ax.set_xlabel("Chrono. Age Group")  # type: ignore
    if x_var == "vo2max_q":
        ax.set_xlabel("VO2Max quintile (within age)")  # type: ignore
    return ax.figure  # type: ignore


def plot_healthy_vs_not_age_hist(datadf, cdf=False):
    """Figure 1 plot"""
    pdf = (
        datadf[datadf.time_bin == "first month"]
        .groupby("canonical_subject_id")
        .agg(
            age=("approx_age", "mean"),
            is_healthy=("is_healthy", "first"),
        )
        .assign(group=lambda x: np.where(x.is_healthy, "'Healthy'", "General"))
    )
    colors = sns.color_palette("colorblind")[0:2]
    fig, ax = plt.subplots(figsize=(10, 4))
    if cdf:
        sns.ecdfplot(
            data=pdf[pdf.age <= 89],
            x="age",
            hue="group",
            stat="proportion",
            palette=colors,
        )
        xlo, xhi = ax.get_xlim()  # type: ignore
        ax.hlines(  # type: ignore
            y=0.5, xmin=xlo, xmax=xhi, color="grey", linestyle="--", alpha=0.5
        )  #  type: ignore
        ax.set_xlim(xlo, xhi)  # type: ignore
    else:
        sns.histplot(
            data=pdf[pdf.age <= 89],
            x="age",
            hue="group",
            element="step",
            bins=50,  # type: ignore
            stat="density",
            common_norm=False,
            alpha=0.1,
            palette=colors,
        )
    ax.set_xlabel("Chrono. Age")  # type: ignore
    return fig, ax


def scatter_with_corr(
    x,
    y,
    xlabel=None,
    ylabel=None,
    plot_y_equal_x=True,
    ax=None,
    alpha=0.1,
    s=5,
    **kwargs,
):
    r, p = sp.stats.pearsonr(x=x, y=y)
    pstr = f"p={np.round(p, 2):.2f}"
    if p < 0.001:
        pstr = "p<.001"
    r2 = r2_score(y_true=y, y_pred=x)
    mae = np.mean(np.abs(y - x))
    rmse = np.std(y - x)
    corr_str = (
        f"pearson r={np.round(r, 2):.2f} ({pstr})\n"  # (p={p:.4g})\n"
        f"r2={np.round(r2, 2):.2f}\n"
        f"mae={np.round(mae, 2):.2f}\n"
        f"rmse={np.round(rmse, 2):.2f}"
    )
    if ax is None:
        _, ax = plt.figure(figsize=(8, 6)), plt.gca()

    ax.scatter(x, y, color="black", s=s, alpha=alpha, **kwargs)  # type: ignore
    t = ax.text(0.75, 0.04, corr_str, transform=ax.transAxes)  # type: ignore
    t.set_bbox(dict(facecolor="white", alpha=1.0))

    if xlabel is not None:
        ax.set_xlabel(xlabel)  # type: ignore

    if ylabel is not None:
        ax.set_ylabel(ylabel)  # type: ignore

    xlim = ax.get_xlim()  # type: ignore
    ylim = ax.get_ylim()  # type: ignore
    if plot_y_equal_x:
        lo = min([xlim[0], ylim[0]])
        hi = max([xlim[1], ylim[1]])
        ax.plot([lo, hi], [lo, hi], "--", c="grey")  # type: ignore
        ax.set_xlim(xlim)  # type: ignore
        ax.set_ylim(ylim)  # type: ignore

    return ax.figure, ax  # type: ignore


def clean_qcut(x, *args):
    return pd.qcut(x, *args).apply(  # type: ignore
        lambda x: pd.Interval(left=round(x.left, 1), right=round(x.right, 1))
    )
