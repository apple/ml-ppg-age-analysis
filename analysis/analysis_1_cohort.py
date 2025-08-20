#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os
import numpy as np
import pandas as pd
from .analysis_util import make_preddf_clean_subset
from .viz import plt, sns, colors


def make_cohort_figure(preddf, subjdf, output_dir):
    ## output describe
    OUTPUT_DIR = os.path.join(output_dir, "fig-cohort")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    figname = lambda x: os.path.join(OUTPUT_DIR, x)

    ## subset to all PPG contributing participants btw 18 and 85
    pdf = make_preddf_clean_subset(
        preddf,
        age_min=18,
        age_max=85,
        model="yhat-mod_global_weight_none-subset-all",
        splits=["train", "test"],
        time_bins=["first month", "last month"],
        min_segments=0,
    )

    ## cohort summary table (makes healthy/general split)
    cohort_table = make_cohort_table(pdf, subjdf)
    cohort_table.to_latex(open(figname("cohort_table.tex"), "w"))
    print(cohort_table)

    ## make plots of duration etc
    make_cohort_plots(subjdf=subjdf, preddf=pdf, figname=figname)


def make_cohort_table(preddf, subjdf):
    """Cohort summary table maker"""
    if "model" in preddf.columns:
        assert preddf.model.nunique() == 1, "only count for one model"

    ## make sure all health related columns are fleshed out
    subj_cols = [
        "canonical_subject_id",
        "height",
        "weight",
        "vo2max_ave",
        "has_smoked",
        "no_meds",
        "no_disease_history",
        "smoker_daily",
        "diabetes",
        "bloodpressure",
        "cholesterol",
        "heartdisease",
        "heartattack",
    ]

    def iqr_str(x):
        xx = x[~np.isnan(x) & ~np.isinf(x)]
        lo, hi = np.nanpercentile(xx, [2.5, 97.5])
        return "%2.1f [%2.1f-%2.1f]" % (np.nanmean(xx), lo, hi)

    def make_table(ddf, label=""):
        subj_sumdf = (
            ddf.groupby(["canonical_subject_id", "time_bin"])
            .agg(
                age=("age", "mean"),
            )
            .reset_index()
            .merge(subjdf[subj_cols], on="canonical_subject_id", how="left")
            .assign(
                bmi=lambda x: x.weight / (x.height / 100) ** 2,
            )
        )
        subj_sumdf = subj_sumdf[subj_sumdf.time_bin == "first month"]
        cf = lambda x: f"{x:,}"
        sumdf = pd.DataFrame(
            {
                "# Subjects": cf(ddf.canonical_subject_id.nunique()),
                "Age": iqr_str(subj_sumdf.age),
                "BMI": iqr_str(subj_sumdf.bmi),
                "VO2Max": iqr_str(subj_sumdf.vo2max_ave),
                "# taking meds": cf(np.sum(~subj_sumdf.no_meds)),
                "# with smoking history": cf(np.sum(subj_sumdf.has_smoked)),
                "# daily smokers": cf(np.sum(subj_sumdf.smoker_daily)),
                "# diabetes": cf(np.sum(subj_sumdf.diabetes == "[yes]")),
                "# hypertension": cf(np.sum(subj_sumdf.bloodpressure == "[yes]")),
                "# heart disease": cf(np.sum(subj_sumdf.heartdisease == "[yes]")),
                "# heart attack": cf(np.sum(subj_sumdf.heartattack == "[yes]")),
            },
            index=[label],
        ).T

        return sumdf

    def make_sex_table(hdf, label):
        h_subj = make_table(hdf, label="All")
        h_male = make_table(hdf[hdf.biological_sex == "Male"], label="Male")
        h_female = make_table(hdf[hdf.biological_sex == "Female"], label="Female")
        h_sum = pd.concat([h_subj, h_female, h_male], axis=1)
        h_sum = pd.concat({label: pd.DataFrame(h_sum)}, axis=1, names=["", ""])
        return h_sum

    # smoker type booleans for counting
    subjdf = subjdf.assign(
        smoker_daily=lambda x: x.smoker_type == "every_day",
        has_smoked=lambda x: ~x.smoker_type.isin(["healthy", "never_smoker"]),
    )

    ## make sure preddf only has one per person
    if "model" in preddf.columns:
        assert preddf.model.nunique() == 1, "only one model per subject here"

    if "age" not in preddf.columns:
        preddf = preddf.assign(age=lambda x: x.approx_age)

    if "biological_sex" not in preddf.columns:
        preddf = preddf.merge(
            subjdf[["canonical_subject_id", "biological_sex"]],
            on="canonical_subject_id",
            how="left",
        )

    healthy_subj = subjdf[subjdf.is_healthy].canonical_subject_id.tolist()
    healthydf = make_sex_table(
        preddf[preddf.canonical_subject_id.isin(healthy_subj)],
        label="'Healthy' subjects",
    )
    # healthydf = make_sex_table(preddf[preddf.is_healthy], label="'Healthy' subjects")
    alldf = make_sex_table(preddf, label="All subjects")
    cohort_table = pd.concat([alldf, healthydf], axis=1)

    return cohort_table


def make_cohort_plots(subjdf, preddf=None, figname=None):
    """Plots and saves cohort histograms"""
    plt.close("all")

    if preddf is not None:
        ## subset to subjects involved in predictions
        subjdf = subjdf[
            subjdf.canonical_subject_id.isin(preddf.canonical_subject_id.unique())
        ]

    print(f"cohort plots: {subjdf.canonical_subject_id.nunique()} subj")

    def iqr_str(x):
        xx = x[~np.isnan(x) & ~np.isinf(x)]
        lo, hi = np.nanpercentile(xx, [25, 75])
        return "%2.1f [%2.1f-%2.1f]" % (np.nanmedian(xx), lo, hi)

    fig, ax = plt.figure(figsize=(6, 2.5)), plt.gca()
    sns.histplot(
        x="n_days_in_study",
        data=subjdf,
        ax=ax,
        alpha=0.5,
        color=colors[0],
        element="step",
    )
    ax.set_xlabel("Days in study")  # type: ignore
    ax.set_ylabel("Subject count")  # type: ignore
    ax.set_xlim(0, subjdf.n_days_in_study.max())  # type: ignore
    current_values = ax.get_yticks()  # type: ignore
    plt.gca().set_yticklabels([f"{x/1000:.0f}k" for x in current_values])  # type: ignore
    ax.set_title(f"Study duration dist: {iqr_str(subjdf.n_days_in_study.values)} days")  # type: ignore
    fig.tight_layout()
    fig.savefig(figname("duration-histogram.pdf"), bbox_inches="tight")
    print(subjdf.n_days_in_study.describe())

    ## subject age + BMI
    pdf = subjdf[(subjdf.age > 18) & (subjdf.age < 85)]

    fig, ax = plt.figure(figsize=(6, 2.5)), plt.gca()
    sns.histplot(
        x="age",
        data=pdf,
        hue="cohort",
        ax=ax,
        bins=40,  # type: ignore
        alpha=0.5,
        common_norm=False,
        stat="probability",
        element="step",
        palette=colors,
    )
    ax.set_xlabel("age (years)")  # type: ignore
    ax.set_ylabel("probability")  # type: ignore
    ax.set_xlim(18, 90)  # type: ignore
    fig.tight_layout()
    fig.savefig(figname("age-histogram.pdf"), bbox_inches="tight")

    fig, ax = plt.figure(figsize=(6, 2.5)), plt.gca()
    pdf = pdf[(subjdf.bmi > 10) & (subjdf.bmi < 100)]
    sns.histplot(
        x="bmi",
        data=pdf,
        hue="cohort",
        ax=ax,
        bins=40,  # type: ignore
        alpha=0.5,
        common_norm=False,
        stat="probability",
        element="step",
        palette=colors,
    )
    ax.set_xlabel("BMI (kg/m^2)")  # type: ignore
    ax.set_ylabel("probability")  # type: ignore
    ax.set_xlim(10, 100)  # type: ignore
    fig.tight_layout()
    fig.savefig(figname("bmi-histogram.pdf"), bbox_inches="tight")

    fig, ax = plt.figure(figsize=(6, 2.5)), plt.gca()
    sns.histplot(
        x="vo2max_ave",
        data=pdf,
        hue="cohort",
        ax=ax,
        bins=40,  # type: ignore
        alpha=0.5,
        common_norm=False,
        stat="probability",
        element="step",
        palette=colors,
    )
    ax.set_xlabel("Estimated VO2Max (mL/kg/min)")  # type: ignore
    ax.set_ylabel("probability")  # type: ignore
    ax.set_xlim(10, 70)  # type: ignore
    fig.tight_layout()
    fig.savefig(figname("vo2max-histogram.pdf"), bbox_inches="tight")
