#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from joblib import Parallel, delayed
from analysis.analysis_util import (
    make_preddf_clean_subset,
    add_per_group_quantiles,
    make_rel_risk_df,
    make_delta_df,
    bootstrap_conf_int,
)

## column names for medical history information
default_medhist_cols = ["diabetes", "cholesterol", "bloodpressure"]


def make_age_gap_medhistdf(
    preddf,
    gap_metric="gap_adj_spline",
    fix_age_gap_buckets=False,
    medhist_cols=default_medhist_cols,
):
    """Make diagnosis rate by age gap quintile dataframes.
    fix_age_gap_buckets: if True, uses fixed age gap buckets e.g., (-inf, -2),
        (-2, 2), (2, 4), (4, 6), (6+) predicted years
    """

    # Subset to relevant subjects + models
    models = [
        "yhat-general-train-mod_global_weight_none-subset-all",
        "yhat-mod_global_weight_none-subset-all",
        "yhat-hr-hrv-baseline",
        "yhat-hr-hrv-rf-baseline",
    ]
    pdf = preddf

    if fix_age_gap_buckets:
        bins = [-np.inf, -2, 0, 2, 4, 6, np.inf]
        pdf = add_per_group_quantiles(
            pdf,
            group_vars=["model"],
            health_metric=gap_metric,
            bins=bins,
        )
        print(pdf[pdf.model == models[1]].metric_group.value_counts())
    else:
        pdf = add_per_group_quantiles(
            pdf,
            group_vars=["biological_sex", "age_bin", "model"],
            health_metric=gap_metric,
        )

    ## print out all quantile bin definitions + save
    bindf = pdf[
        (pdf.model == models[1])
        & (pdf.biological_sex.isin(["Male", "Female"]))
        & (pdf.age_bin != ">75")
    ][
        [
            "biological_sex",
            "age_bin",
            "vo2max_q",
            "vo2max_group",
            "metric_q",
            "metric_group",
        ]
    ].drop_duplicates()

    ## for each condition, make plots and save summary dataframes
    rrdfs = [
        make_rel_risk_df(
            preddf=pdf[pdf.time_bin == "first month"],
            condition=c,
            groups=["age_bin", "biological_sex", "model"],
            health_metric=gap_metric,
            seed=0,
        )
        for c in medhist_cols
    ]
    rrdf = pd.concat(rrdfs, axis=0)  # type: ignore

    ## also average over age_bin
    rrdfs_noage = Parallel(n_jobs=46, verbose=10)(
        delayed(make_rel_risk_df)(
            preddf=pdf,
            condition=c,
            groups=["biological_sex", "model"],
            health_metric=gap_metric,
            seed=0,
        )
        for c in medhist_cols
    )
    rrdfs_noage = pd.concat(rrdfs_noage, axis=0)  # type: ignore
    print(
        rrdfs_noage[
            (rrdfs_noage.condition.isin(["diabetes", "heartdisease", "heartfailure"]))
            & (rrdfs_noage.variable == "rel_risk")
            & (rrdfs_noage.model == models[1])
        ][["condition", "biological_sex", "ci_str", "metric_group"]]
    )

    ## also stratify by within group VO2Max
    rrdfs_vo2max = Parallel(n_jobs=46, verbose=10)(
        delayed(make_rel_risk_df)(
            preddf=pdf,
            condition=c,
            groups=["age_bin", "vo2max_q", "biological_sex", "model"],
            health_metric=gap_metric,
            seed=0,
        )
        for c in medhist_cols
    )
    rrdf_vo2max = pd.concat(rrdfs_vo2max, axis=0)  # type: ignore

    return rrdf, rrdf_vo2max, bindf


def make_age_rate_medhistdf(
    preddf,
    subjdf,
    model="yhat-mod_global_weight_none-subset-all",
    medhist_cols=default_medhist_cols,
):
    pdf = preddf[preddf.model == model]
    deltadf = make_delta_df(preddf, min_years=0.5).merge(
        subjdf[["canonical_subject_id", "vo2max_ave"] + medhist_cols],
        on="canonical_subject_id",
        how="left",
    )
    deltadf = add_per_group_quantiles(
        deltadf,
        group_vars=["biological_sex", "age_bin"],
        health_metric="gap_adj_spline",
    ).rename(columns={"metric_q": "age_gap_q"})
    deltadf = add_per_group_quantiles(
        deltadf,
        group_vars=["biological_sex", "age_bin", "age_gap_q"],
        health_metric="gap_rate",
    )  # .rename(columns={"metric_q": "age_rate_q"})
    deltadf = deltadf.assign(split="test", time_bin="first month")

    ## for each condition, make plots and save summary dataframes
    rrdfs = Parallel(n_jobs=46, verbose=10)(
        delayed(make_rel_risk_df)(
            preddf=deltadf,
            condition=c,
            groups=["age_bin", "age_gap_q", "biological_sex", "model"],
            health_metric="gap_rate",
            seed=0,
        )
        for c in medhist_cols
    )
    rrdf = pd.concat(rrdfs, axis=0)  # type: ignore
    return rrdf


def make_years_above_healthy_df(
    preddf,
    subjdf,
    time_bins=["first month"],
    outcome="smoking_status",
):
    """Make "years above healthy" dataframe for ggplot plotting.
    outcome = "smoking_status"  # "alcohol_volume" #frequency" #, #'alcohol_ever' #ealth_behavior_alcohol_q1b'
    subset to regdf with n days > 365
    """

    # subset to model
    pdf = make_preddf_clean_subset(
        preddf,
        age_min=18,
        age_max=85,
        splits=["train", "test"],
        min_segments=30,
        time_bins=time_bins,
    )

    # compute age/sex matched healthy age gaps
    healthy_pdf = (
        pdf[pdf.is_healthy]
        .groupby(["age_bin", "biological_sex"])
        .agg(
            healthy_gap=("gap", "mean"),
            healthy_gap_adj_spline=("gap_adj_spline", "mean"),
            n_healthy=("gap", len),
        )
        .reset_index()
    )
    print(healthy_pdf)
    pdf = pdf.merge(
        healthy_pdf,
        on=["age_bin", "biological_sex"],
        how="left",
    ).assign(
        gap_adj=lambda x: x.gap - x.healthy_gap,
        # gap_adj_spline=lambda x: x.gap_adj_spline-x.healthy_gap_adj_spline,
    )

    pdf = pdf[
        (pdf.smoking_status != "n/a") & (pdf.smoking_status != "prefer_not_to_answer")
    ].assign(
        smoker_type=lambda x: pd.Categorical(
            x.smoker_type,
            categories=["never_smoker", "not_at_all", "some_days", "every_day"],
        )
    )

    print(pdf.drop_duplicates("canonical_subject_id")["smoking_status"].value_counts())
    ci_fun = lambda x: bootstrap_conf_int(x.values)

    def summarize_col_(pdf, col="gap_adj"):
        years_added_df = (
            pdf.groupby(["age_bin", "biological_sex", "smoking_status"])[col]
            .apply(ci_fun)
            .reset_index()
            .rename(columns={"level_3": "var"})
        )
        years_added_df_all = (
            pdf.groupby(["age_bin", "smoking_status"])[col]
            .apply(ci_fun)
            .reset_index()
            .assign(biological_sex="All")
            .rename(columns={"level_2": "var"})
        )
        years_added_df_grouped = (
            pdf.groupby(["smoking_status"])[col]
            .apply(ci_fun)
            .reset_index()
            .assign(biological_sex="All", age_bin="All")
            .rename(columns={"level_1": "var"})
        )
        years_added_df_sex_grouped = (
            pdf.groupby(["biological_sex", "smoking_status"])[col]
            .apply(ci_fun)
            .reset_index()
            .assign(age_bin="All")
            .rename(columns={"level_2": "var"})
        )
        return pd.concat(
            [
                years_added_df,
                years_added_df_all,
                years_added_df_grouped,
                years_added_df_sex_grouped,
            ],
            axis=0,
        )

    years_added_df = summarize_col_(pdf, col="gap_adj")
    years_added_df_adj = summarize_col_(pdf, col="gap_adj_spline")

    htdf = (
        pdf.groupby(["age_bin", "biological_sex", "bloodpressure", "smoking_status"])[
            "gap_adj_spline"
        ]
        .apply(ci_fun)
        .reset_index()
        .rename(columns={"level_4": "var"})
    )
    print(
        htdf[
            htdf.bloodpressure.isin(["[no]", "[yes]"])
            & (htdf.biological_sex == "Male")
            & (htdf.smoking_status.isin(["every_day", "some_days", "never_smoker"]))
            & (htdf["var"] == "b_ci_str")
        ]
    )

    ## make analysis df
    adf = pdf[(pdf.biological_sex == "Male") & (pdf.age_bin != ">75")]  # .assign(
    #    age_bin=lambda x: x.age_bin.cat.remove_unused_categories()
    # )
    print(adf.columns)

    mod = ols(
        formula=(
            "gap_adj_spline ~ "
            " C(age_bin, Treatment(reference='<25'))"
            " + C(bloodpressure, Treatment(reference='[no]')) "
            " + C(diabetes, Treatment(reference='[no]')) "
            " + C(cholesterol, Treatment(reference='[no]')) "
            " + C(smoking_status, Treatment(reference='healthy'))"
        ),  #: + bloodpressure +  smoking_status",
        data=adf,
    )
    fit = mod.fit()
    fit.summary()

    # get ratedf
    deltadf = make_delta_df(preddf, min_years=1.0)
    deltadf = deltadf.merge(
        subjdf[["canonical_subject_id", "smoking_status", "smoker_type"]],
        on="canonical_subject_id",
        how="left",
    )
    deltadf = deltadf[
        (deltadf.smoking_status != "n/a")
        & (deltadf.smoking_status != "prefer_not_to_answer")
        & deltadf.slope_adj.notnull()
    ].assign(
        smoker_type=lambda x: pd.Categorical(
            x.smoker_type,
            categories=["never_smoker", "not_at_all", "some_days", "every_day"],
        )
    )
    deltadf = deltadf[deltadf.age_bin != ">75"]  # .assign(
    #    age_bin=lambda x: x.age_bin.cat.remove_unused_categories(),
    # )
    return years_added_df, years_added_df_adj
