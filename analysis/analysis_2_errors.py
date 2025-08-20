#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os
from .analysis_util import make_preddf_clean_subset
import numpy as np
import pandas as pd
from . import viz
from .viz import plt, sns, colors


def make_error_figure(preddf, output_dir):
    ## output
    OUTPUT_DIR = os.path.join(output_dir, "fig-errors")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    figname = lambda x: os.path.join(OUTPUT_DIR, x)

    #############################
    ### compare all models     ##
    #############################
    pdf = make_preddf_clean_subset(
        preddf,
        age_min=18,
        age_max=85,
        splits=["train", "test"],
        min_segments=30,
        model=[
            "yhat-general-train-mod_global_weight_none-subset-all",
            "yhat-mod_global_weight_none-subset-all",
            "yhat-hr-hrv-baseline",
            "yhat-hr-hrv-rf-baseline",
            "yhat-hr-hrv-minimal-rf-baseline",
        ],  ## type: ignore
    )

    print(pdf.model.value_counts())
    errdf_global = pd.concat(
        [
            compute_errordf(
                preddf=pdf, groups=["split", "biological_sex", "is_healthy", "model"]
            ),
            compute_errordf(preddf=pdf, groups=["split", "model", "is_healthy"]).assign(
                biological_sex="All"
            ),
        ],
        axis=0,
    )
    print(errdf_global)
    errdf_global.to_csv(figname("error-global.csv"))

    ## for different demographic categories, also compute errors
    grps = ["race_eth", "bmiq", "age_bin", "biological_sex"]
    for g in grps:
        edf = compute_errordf(
            preddf=pdf,
            groups=["split", "model", "is_healthy"] + [g],
        )
        edf.to_csv(figname(f"error-{g}.csv"))

    ## make and save prediction plots (y vs yhat), demographic error plots
    pdf = make_preddf_clean_subset(
        preddf,
        age_min=18,
        age_max=85,
        splits=["test"],
        min_segments=30,
        model="yhat-mod_global_weight_none-subset-all",
    )
    make_prediction_plots(pdf, figname)
    make_demographic_plots(pdf, figname)


def compute_errordf(preddf, groups=["split"]):
    """Compute summary of error by group"""

    def ci_str(
        x,
        fun=lambda x: np.mean(np.abs(x)),
        n_boot=1000,
        return_lo=False,
        return_hi=False,
    ):
        rs = np.random.RandomState(0)
        bs = []
        for _ in range(n_boot):
            idx = rs.choice(len(x.values), size=len(x.values))
            bs += [fun(x.values[idx])]
        ## total statistics
        b_total = fun(x.values)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        if return_lo:
            return lo
        if return_hi:
            return hi
        return f"{b_total:.2f} [{lo:.2f} - {hi:.2f}]"

    health_errordf = (
        preddf.groupby(groups)
        .agg(
            age_mae=("gap", lambda x: np.mean(np.abs(x))),
            age_mae_lo=("gap", lambda x: ci_str(x, return_lo=True)),
            age_mae_hi=("gap", lambda x: ci_str(x, return_hi=True)),
            age_mae_str=("gap", ci_str),
            age_bias=("gap", "mean"),
            n_subj=("canonical_subject_id", lambda x: x.nunique()),
            n_obj=("gap", len),
        )
        .reset_index()
    )
    return health_errordf


def make_prediction_plots(preddf, figname):
    models = preddf.model.unique()

    # is_healthy=True
    # model = models[0]
    def plot_preds(is_healthy, model, sex):
        pdf = preddf[
            (preddf.model == model)
            & (preddf.split == "test")
            & (preddf.is_healthy == is_healthy)
        ]
        if sex != "All":
            pdf = pdf[pdf.biological_sex == sex]
        fig, ax = plt.figure(figsize=(5, 4)), plt.gca()
        nsubj = pdf.canonical_subject_id.nunique()
        ax.set_xlim(10, 85)  # type: ignore
        ax.set_ylim(10, 85)  # type: ignore
        if is_healthy:
            alpha = 0.2
        else:
            alpha = 0.02
        _ = viz.scatter_with_corr(pdf.value, pdf.age, ax=ax, alpha=alpha)
        ax.set_ylabel("Chrono. Age")  # type: ignore
        ax.set_xlabel("Predicted Age")  # type: ignore
        if is_healthy:
            ax.set_title(f"Healthy cohort, {sex.lower()} (n={nsubj:,})")  # type: ignore
        else:
            ax.set_title(f"General cohort, {sex.lower()} (n={nsubj:,})")  # type: ignore
        fig.tight_layout()
        return fig, ax

    plt.close("all")
    for model in models:
        print("Plotting model: ", model)
        for is_healthy in [True, False]:
            for sex in ["All", "Male", "Female"]:
                fig, _ = plot_preds(is_healthy, model, sex)
                fig.savefig(
                    figname(f"preds-{model}-healthy-{is_healthy}-{sex}.pdf"),
                    bbox_inches="tight",
                )
                fig.savefig(
                    figname(f"preds-{model}-healthy-{is_healthy}-{sex}.png"),
                    bbox_inches="tight",
                    dpi=300,
                )


def make_demographic_plots(preddf, figname):
    def barplot(hue="race_eth", legend_title="Self report race/eth", width=6):
        fig, ax = plt.figure(figsize=(width, 3)), plt.gca()
        ax = sns.barplot(
            data=preddf,
            x="cohort",
            y="gap",
            hue=hue,
            estimator=lambda x: np.mean(np.abs(x)),  # type: ignore
            ax=ax,
            palette=colors,
            dodge=0.4,  # type: ignore
        )
        hatches = ["//", "\\\\", "-", "x", "+", "."]
        handles = []
        for bars, hatch, handle in zip(
            ax.containers, hatches, ax.get_legend_handles_labels()[0]  # type: ignore
        ):
            handle.set_hatch(hatch)
            for bar in bars:
                bar.set_hatch(hatch)
            handles += [handle]
        ax.legend(  # type: ignore
            bbox_to_anchor=(1.01, 0.75),
            loc="upper left",
            title=legend_title,
            handles=handles,
        )
        ax.set_ylim(0, 4.5)  # type: ignore
        ax.set_xlabel("Cohort")  # type: ignore
        ax.set_ylabel("MAE (years)")  # type: ignore
        fig.tight_layout()
        return fig, ax

    sns.set_style("whitegrid")
    plt.close("all")

    fig, _ = barplot(hue="race_eth", legend_title="Race/Eth (self report)")
    fig.savefig(figname("mae-by-race.pdf"), bbox_inches="tight")

    fig, _ = barplot(hue="bmiq", legend_title="BMI ($kg/m^2$)")
    fig.savefig(figname("mae-by-bmi.pdf"), bbox_inches="tight")

    fig, _ = barplot(hue="biological_sex", legend_title="Bio. Sex (self report)")
    fig.savefig(figname("mae-by-sex.pdf"), bbox_inches="tight")

    fig, _ = barplot(hue="age_bin", legend_title="Chron. Age")
    fig.savefig(figname("mae-by-age.pdf"), bbox_inches="tight")
