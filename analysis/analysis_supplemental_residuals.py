#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os
from analysis.analysis_util import make_preddf_clean_subset, make_delta_df
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot, norm
from statsmodels.graphics.agreement import mean_diff_plot


def make_supp_residual_fig(preddf, output_dir):
    ## save figures to output dir
    OUTPUT_DIR = os.path.join(output_dir, "fig-supp-residual-analysis")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    figname = lambda x: os.path.join(OUTPUT_DIR, x)

    ### subset to just the one model
    pdf = make_preddf_clean_subset(
        preddf,
        age_min=18,
        age_max=85,
        model="yhat-mod_global_weight_none-subset-all",
        splits=["test"],
        time_bins=["first month"],  # "last month"],
        min_segments=30,
    )
    cohort_subj = pdf.canonical_subject_id.unique().tolist()
    print(len(cohort_subj))

    ## bland altman plot
    make_bland_altman_plot(pdf, figname)

    ## QQ and outlier analysis
    make_qq_plot(pdf, figname)

    ## delta age gap plots
    make_age_gap_delta_plot(preddf, figname, min_years=0.9)


def make_age_gap_delta_plot(
    preddf,
    figname,
    min_years=2.0,
):
    deltadf = make_delta_df(preddf, min_years=min_years, gap_metric="gap_adj_spline")

    def compare_rates(deltadf, ax, sex="Male"):
        df = deltadf[(deltadf.biological_sex == sex) & ~deltadf.age_bin.isin([">75"])]
        df = df.assign(
            # age_bin=lambda x: x.age_bin.cat.remove_unused_categories(),
            cohort=lambda x: np.where(x.is_healthy, "healthy", "general"),
        )
        cnts = df.cohort.value_counts()
        sns.pointplot(
            x="age_bin",
            y="aging_rate",
            hue="cohort",
            data=df,
            ax=ax,
            dodge=0.2,  # type: ignore
        )
        ax.set_xlabel("Chron. age group")
        ax.set_ylabel("PpgAge rate")
        ax.set_title(
            f"PpgAge Rate, {sex}\nn={cnts.loc['general']:,} general and n={cnts.loc['healthy']:,} healthy"
        )

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10, 3.0))
    compare_rates(deltadf, ax=axa, sex="Male")
    compare_rates(deltadf, ax=axb, sex="Female")
    fig.tight_layout()
    fig.savefig(figname("ppgage-delta.pdf"), bbox_inches="tight")


def make_qq_plot(pdf, figname):
    def qq_resid(ys, yhat, ax):
        resids = yhat - ys
        zs = resids / np.std(resids)
        (osm, osr), (slope, intercept, r) = probplot(zs)
        ax.plot(osm, osr, marker=".")
        xlo, xhi = ax.set_xlim()
        ylo, yhi = ax.set_ylim()
        lo, hi = min([xlo, ylo]), max([xhi, yhi])
        ax.plot(
            [lo, hi], [lo, hi], label="Theoretical Normal", linestyle="--", color="grey"
        )
        ax.legend()
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Normal quantiles")
        ax.set_ylabel("Ordered values")
        ax.set_title("QQ Plot")

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8, 3.2))
    ## healthy
    testdf = pdf[(pdf.split == "test") & pdf.is_healthy]
    ys = testdf.age.values
    yhat = testdf.value.values
    qq_resid(ys, yhat, ax=axa)
    axa.set_title(f"QQ plot, n={len(ys):,} healthy participants")
    ## general
    testdf = pdf[(pdf.split == "test") & ~pdf.is_healthy]
    ys = testdf.age.values
    yhat = testdf.value.values
    qq_resid(ys, yhat, ax=axb)
    axb.set_title(f"QQ plot, n={len(ys):,} general participants")
    ## save out
    fig.tight_layout()
    fig.savefig(figname("qq-plots.pdf"), bbox_inches="tight")
    fig.savefig(figname("qq-plots.png"), bbox_inches="tight", dpi=600)

    ## plot residual distributions
    def resid_hist(ys, yhat, ax):
        resids = yhat - ys
        n, bins, _ = ax.hist(resids, bins=50)
        xnorm = (
            norm.pdf(bins, loc=resids.mean(), scale=resids.std())
            * np.sum(n)
            * np.diff(bins)[0]
        )
        ax.plot(bins, xnorm, color="grey", label="normal dist.")
        ax.legend()
        ax.set_xlabel("PpgAge Gap")
        ax.set_ylabel("Frequency (# participants)")

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8, 2.75))
    testdf = pdf[(pdf.split == "test") & pdf.is_healthy]
    ys = testdf.age.values
    yhat = testdf.value.values
    resid_hist(ys, yhat, ax=axa)
    axa.set_title(f"PpgAge Residuals, Healthy Cohort (n={len(yhat):,})")
    testdf = pdf[(pdf.split == "test") & ~pdf.is_healthy]
    ys = testdf.age.values
    yhat = testdf.value.values
    resid_hist(ys, yhat, ax=axb)
    axb.set_title(f"PpgAge Residuals, General Cohort (n={len(yhat):,})")
    fig.tight_layout()
    fig.savefig(figname("residual-histograms.pdf"), bbox_inches="tight")


def make_bland_altman_plot(pdf, figname):
    """Save bland altman plot for healthy and general test participants"""

    # healthy and general side by side
    fig, axarr = plt.subplots(1, 2, figsize=(10, 2.75))

    ## healthy
    testdf = pdf[(pdf.split == "test") & pdf.is_healthy]
    n_subj = testdf.canonical_subject_id.nunique()
    ys = testdf.age.values
    yhat = testdf.value.values
    ax = axarr[0]
    mean_diff_plot(ys, yhat, ax=ax, scatter_kwds=dict(alpha=0.5, s=2, c="grey"))
    ax.set_title(f"Healthy (n={n_subj:,} test participants)")

    ## general
    testdf = pdf[(pdf.split == "test") & (pdf.is_healthy == False)]
    n_subj = testdf.canonical_subject_id.nunique()
    ys = testdf.age.values
    yhat = testdf.value.values
    ax = axarr[1]
    mean_diff_plot(ys, yhat, ax=axarr[1], scatter_kwds=dict(alpha=0.2, s=2, c="grey"))
    ax.set_title(f"General (n={n_subj:,} test participants)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(figname("bland-altman-plot.pdf"), bbox_inches="tight")
    fig.savefig(figname("bland-altman-plot.png"), bbox_inches="tight", dpi=600)
