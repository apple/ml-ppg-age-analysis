#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import numpy as np
import pandas as pd
from analysis.analysis_util import bucket_age


######################################################
# data loaders for downstream analyses (see readme)  #
######################################################


def load_segment_predictions():
    rs = np.random.RandomState(0)

    def makedf_(model, err):
        n_subj = 1000
        dfs = []
        opts = ["[yes]", "[no]"]
        timegrid = pd.date_range(start="2024-01-01", end="2024-12-31", freq="12h")
        for subj in range(n_subj):
            start_age = rs.uniform(low=20, high=80)
            age = start_age + (timegrid - timegrid[0]).days / 365.25
            sdf = pd.DataFrame(
                {
                    "canonical_subject_id": subj,
                    "local_start_time": timegrid,
                    "local_date": timegrid.date,
                    "age": age,
                    "model": model,
                    "value": age + err * rs.randn(*age.shape),
                    "biological_sex": rs.choice(["Male", "Female"]),
                    "vo2max_ave": rs.uniform(low=20, high=60),
                    "race_eth": rs.choice(
                        [
                            "Asian",
                            "Black",
                            "Hispanic",
                            "Middle Eastern",
                            "White",
                            "other",
                        ]
                    ),
                    "bloodpressure": rs.choice(opts),
                    "diabetes": rs.choice(opts),
                    "cholesterol": rs.choice(opts),
                    "heartdisease": rs.choice(opts),
                    "heartattack": rs.choice(opts),
                    "smoker_type": rs.choice(
                        [
                            "healthy",
                            "never_smoker",
                            "not_at_all",
                            "some_days",
                            "every_day",
                        ]
                    ),
                    "split": "test",
                    "is_healthy": rs.choice([True, False]),
                    "weight": rs.uniform(low=40, high=136),
                    "height": rs.uniform(1.4, high=2.1),
                    "no_meds": rs.choice([True, False]),
                    "no_disease_history": rs.choice([True, False]),
                    "cohort": rs.choice(["healthy", "general"]),
                }
            ).assign(
                month=lambda x: x.local_start_time.dt.month,
                time_bin=lambda x: np.where(
                    x.month == 1,
                    "first month",
                    np.where(x.month == 12, "last month", "mid"),
                ),
                smoking_status=lambda x: x.smoker_type,
                bmi=lambda x: x.weight / x.height**2,
            )
            dfs += [sdf]
        return pd.concat(dfs, axis=0)

    # make our model
    single_df = makedf_(
        model="yhat-general-train-mod_global_weight_none-subset-all", err=8
    )

    ## make other comparison models
    yhatdf = pd.concat(
        [
            single_df,
            single_df.assign(
                model="yhat-mod_global_weight_none-subset-all",
                value=lambda x: x.age + 12 * rs.randn(*x.age.shape),
            ),
            single_df.assign(
                model="yhat-hr-hrv-baseline",
                value=lambda x: x.age + 15 * rs.randn(*x.age.shape),
            ),
            single_df.assign(
                model="yhat-hr-hrv-rf-baseline",
                value=lambda x: x.age + 15 * rs.randn(*x.age.shape),
            ),
        ]
    )

    return yhatdf


def make_subject_month_predictions(yhatdf):
    ## compute n days in study by subj
    ndaysdf = (
        yhatdf.groupby("canonical_subject_id")
        .agg(
            n_days_in_study=("local_start_time", lambda x: (x.max() - x.min()).days),
        )
        .reset_index()
    )
    preddf = (
        yhatdf.groupby(["canonical_subject_id", "time_bin", "model"])
        .agg(
            value=("value", "mean"),
            age=("age", "mean"),
            biological_sex=("biological_sex", "first"),
            race_eth=("race_eth", "first"),
            vo2max_ave=("vo2max_ave", "first"),
            diabetes=("diabetes", "first"),
            bloodpressure=("bloodpressure", "first"),
            cholesterol=("cholesterol", "first"),
            smoking_status=("smoking_status", "first"),
            smoker_type=("smoker_type", "first"),
            split=("split", "first"),
            is_healthy=("is_healthy", "first"),
            height=("height", "first"),
            weight=("weight", "first"),
            no_meds=("no_meds", "first"),
            heartdisease=("heartdisease", "first"),
            heartattack=("heartattack", "first"),
            no_disease_history=("no_disease_history", "first"),
            bmi=("bmi", "first"),
            cohort=("cohort", "first"),
            n_segments=("value", len),
        )
        .assign(
            gap=lambda x: x.value - x.age,
            gap_adj_spline=lambda x: x.gap,
            age_bin=lambda x: bucket_age(x.age),
        )
        .reset_index()
        .merge(ndaysdf, on="canonical_subject_id", how="left")
    )

    return preddf


def make_dailydf(yhatdf):
    dailydf = (
        yhatdf[yhatdf.model == "yhat-mod_global_weight_none-subset-all"]
        .groupby(["canonical_subject_id", "local_date"])
        .agg(
            yhat_mean=("value", "mean"),
            age=("age", "mean"),
            bmi=("bmi", "mean"),
        )
        .reset_index()
        .assign(local_date=lambda x: pd.to_datetime(x.local_date))
        .rename(columns={"yhat_mean": "yhat-mean"})
        .set_index("canonical_subject_id")
    )
    return dailydf


def make_pregnancy_outcome_df(subjdf):
    rs = np.random.RandomState(0)
    preg_outcome_df = subjdf.assign(
        pregnancy_outcome_date=pd.to_datetime(rs.choice(["2024-9-02", None])),
        pregnancy_outcome_type=rs.choice(["cesarean section", "vaginal delivery"]),
        pregnancy_dm=rs.choice([True, False]),
        preeclampsia=rs.choice([True, False]),
    )
    return preg_outcome_df


def make_survival_outcome_df(subjdf):
    rs = np.random.RandomState(0)
    survdf = subjdf.copy()
    N = survdf.shape[0]
    ## get event status and event/censor dates
    survdf["is_event"] = rs.choice([1.0, 0.0], N)
    survdf["end_date"] = rs.uniform(.25, 5, N)
    ## build out the covariates to be used in survival modeling
    survdf["hx_bp"] = (survdf["bloodpressure"]=="[yes]").astype(float)
    survdf["hx_chol"] = (survdf["cholesterol"]=="[yes]").astype(float)
    survdf["hx_dm"] = (survdf["diabetes"]=="[yes]").astype(float)
    survdf["smoke_everyday"] = (survdf["smoker_type"]=="every_day").astype(float)
    survdf["smoke_somedays"] = (survdf["smoker_type"]=="some_days").astype(float)
    survdf["smoke_past"] = (survdf["smoker_type"]=="not_at_all").astype(float)
    survdf["sex_is_female"] = (survdf["biological_sex"]=="Female").astype(float)

    survdf = survdf[["canonical_subject_id", "is_event", "end_date", 
                     "age", "gap_adj_spline",
                     "sex_is_female", "vo2max_ave", "bmi",
                     "hx_bp", "hx_chol", "hx_dm", 
                     "smoke_everyday", "smoke_somedays","smoke_past"]]

    return survdf