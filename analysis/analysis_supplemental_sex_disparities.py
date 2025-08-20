#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from analysis.analysis_util import make_preddf_clean_subset


def make_sex_disparity_fig(preddf, subjdf, output_dir):
    ## save figures to output dir
    OUTPUT_DIR = os.path.join(output_dir, "fig-supp-sex-disparities")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    figname = lambda x: os.path.join(OUTPUT_DIR, x)

    ### subset to just the one model
    pdf = make_preddf_clean_subset(
        preddf,
        age_min=18,
        age_max=85,
        model="yhat-mod_global_weight_none-subset-all",
        splits=["train", "test"],
        time_bins=["first month"],
        min_segments=30,
    )
    cohort_subj = pdf.canonical_subject_id.unique().tolist()
    print(len(cohort_subj))

    ## breakdown of biological sex distribution by health category and medical history
    sex_disparities_by_healthy_category(pdf, subjdf, figname)
    sex_disparities_by_medical_history(pdf, subjdf, figname)


def sex_disparities_by_healthy_category(pdf, subjdf, figname):
    ## create one-row-per-subject summary
    sdf = pdf[["canonical_subject_id", "age", "age_bin", "value"]].merge(
        subjdf.drop(columns=["age", "age_bin", "value"], errors="ignore"),
        on="canonical_subject_id",
        how="left",
    )

    def sumdf_(subdf, grp="all"):
        N = subdf.canonical_subject_id.nunique()
        sumdf = (
            subdf[["biological_sex"]]
            .value_counts(normalize=True)
            .reset_index()
            .rename(columns={"proportion": "frac"})
            .assign(group=f"{grp}\n(n={N:,})", total=N)
        )
        return sumdf

    def make_compdf(sdf):
        sumdf = pd.concat(
            [
                sumdf_(sdf),
                sumdf_(sdf[sdf.smoker_type == "never_smoker"], grp="never smoker"),
                sumdf_(sdf[sdf.no_meds], grp="no meds"),
                sumdf_(sdf[sdf.no_disease_history], grp="no disease"),
            ],
            axis=0,
        )
        print(sumdf_(sdf).columns.tolist())
        sumdf = sumdf.assign(percent=lambda x: x.frac * 100.0)
        return sumdf

    def plot_comps(sumdf, title=None):
        fig, ax = plt.figure(figsize=(8, 3)), plt.gca()
        sns.barplot(
            x="group",
            y="percent",
            hue="biological_sex",
            hue_order=["Female", "Male"],
            data=sumdf,
        )
        ax.set_ylabel("Percent")  # type: ignore
        ax.legend(ncol=2)  # type: ignore
        if title is not None:
            ax.set_title(title)  # type: ignore
        fig.tight_layout()
        return fig, ax

    sumdf = make_compdf(sdf)
    fig, _ = plot_comps(sumdf)
    fig.savefig(figname("sex_disparities_by_group.pdf"), bbox_inches="tight")
    sumdf.to_csv(figname("sex_disparities_by_group.csv"))

    age_bins = sdf.age_bin.unique()  # sdf.age_bin.cat.categories
    for age_bin in age_bins:
        adf = make_compdf(sdf[sdf.age_bin == age_bin])
        fig, _ = plot_comps(adf, title=f"Age {age_bin}")
        fig.savefig(
            figname(f"sex_disparities_by_group-{age_bin}.pdf"),
            bbox_inches="tight",
        )
        adf.to_csv(figname(f"sex_disparities_by_group-{age_bin}.csv"))


def sex_disparities_by_medical_history(
    pdf,
    subjdf,
    figname,
    medhist_cols=["diabetes", "bloodpressure", "cholesterol"],
):
    ## create one-row-per-subject summary
    sdf = pdf[["canonical_subject_id", "age", "age_bin"]].merge(
        subjdf.drop(columns=["age", "age_bin"], errors="ignore"),
        on="canonical_subject_id",
        how="left",
    )

    conddf = (
        sdf[sdf.biological_sex.isin(["Male", "Female"])]
        .melt(
            id_vars=["canonical_subject_id", "biological_sex", "age_bin"],
            value_vars=medhist_cols,  # type: ignore
            value_name="metric_value",
        )
        .assign(
            is_no=lambda x: x.metric_value == "[no]",
            is_dunno=lambda x: (x.metric_value == "[do_not_know]")
            | (x.metric_value == "[dont_know]"),
        )
        .dropna()
    )

    ## Determine Order of Conditions
    sumdf = (
        conddf.groupby(["biological_sex", "variable"])
        .agg(frac_no=("metric_value", lambda x: np.mean(x == "[no]")))
        .reset_index()
    )
    sumdf.pivot_table(index="variable", values=["frac_no"], columns=["biological_sex"])
    orderdf = (
        conddf.groupby("variable")
        .agg(
            frac_no=("is_no", "mean"),
            frac_dunno=("is_dunno", "mean"),
        )
        .sort_values("frac_no")
    )

    def plot_frac_no_by_condition(sdf, age_bin=None):
        pdf = sdf
        if age_bin is not None:
            pdf = sdf[sdf.age_bin == age_bin]
        fig, ax = plt.figure(figsize=(12, 4)), plt.gca()
        sns.barplot(
            x="variable",
            y="is_no",
            hue="biological_sex",
            order=orderdf.index.tolist(),
            linestyle="none",
            data=pdf,
            estimator="mean",
            n_boot=10,
            hue_order=["Female", "Male"],
        )
        plt.xticks(rotation=45)
        if age_bin is not None:
            ax.set_title(f"age bin {age_bin}")  # type: ignore
        ax.set_xlabel("Condition")  # type: ignore
        ax.set_ylabel("Fraction 'no'")  # type: ignore
        fig.tight_layout()
        return fig, ax

    fig, _ = plot_frac_no_by_condition(conddf, age_bin=None)
    fig.savefig(figname("condition-all-ages.pdf"), bbox_inches="tight")

    age_bins = sdf.age_bin.unique()  # sdf.age_bin.cat.categories
    for ab in age_bins:
        fig, _ = plot_frac_no_by_condition(conddf, age_bin=ab)
        fig.savefig(figname(f"condition-age-{ab}.pdf"), bbox_inches="tight")
