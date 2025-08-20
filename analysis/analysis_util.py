#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.gam.api import GLMGam, BSplines


def make_preddf_clean_subset(
    preddf,
    age_min=25,
    age_max=75,
    model="yhat-mod_global_weight_none-subset-all",
    splits=["test"],
    min_segments=30,
    time_bins=["first month"],
):
    pdf = preddf[
        (preddf.age >= age_min)
        & (preddf.age <= age_max)
        & (preddf.time_bin.isin(time_bins))
        & preddf.split.isin(splits)
    ]
    if isinstance(model, list):
        print("  subsetting to multiple models!")
        pdf = pdf[pdf["model"].isin(model)]
    else:
        pdf = pdf[pdf.model == model]

    pdf = pdf.assign(
        ape=lambda x: np.abs(x.gap) / x.age,
        cohort=lambda x: np.where(x.is_healthy, "healthy", "general"),
        bmiq=lambda x: pd.cut(
            x.bmi,
            [0, 18.5, 25, 30, np.inf],
            labels=["<18.5", "[18.5-25)", "[25-30)", ">=30"],
            right=False,
        ),
        # age_bin=lambda x: pd.Categoricalx.age_bin.cat.remove_unused_categories(),
    )
    pdf = pdf[pdf.biological_sex.isin(["Male", "Female"])]
    pdf = pdf[pdf.n_segments >= min_segments]
    pdf = pdf.assign(
        race_eth_clean=lambda x: np.where(
            x.race_eth.isin(["other", "Middle Eastern"]), "other", x.race_eth
        ),
        race_eth=lambda x: pd.Categorical(
            x.race_eth_clean, categories=sorted(x.race_eth_clean.unique())
        ),
    )
    return pdf


def make_delta_df(
    preddf,
    min_years=0.5,
    gap_metric="gap_adj_spline",
    time_bins=["first month", "last month"],
):
    """make a dataframe taht mimics preddf, but uses aging rate (estimated
    from last month - first month"""

    pdf = make_preddf_clean_subset(
        preddf,
        age_min=18,
        age_max=85,
        splits=["train", "test"],
        min_segments=30,
        time_bins=time_bins,
    )
    cntdf = pdf.groupby("canonical_subject_id").agg(n=("age", len))
    full_subj = cntdf[cntdf.n == 2].index.unique().tolist()
    pdf = pdf[pdf.canonical_subject_id.isin(full_subj)]
    print(pdf)

    assert pdf.model.nunique() == 1, "only one model in delta df for now"

    # compute gap and predicted rate per subject
    pdf = (
        pdf.sort_values("time_bin")
        .groupby("canonical_subject_id")
        .agg(
            pred_age_delta=("value", lambda x: x.values[1] - x.values[0]),
            age_delta=("age", lambda x: x.values[1] - x.values[0]),
            gap_delta=(gap_metric, lambda x: x.values[1] - x.values[0]),
            gap=(gap_metric, "first"),
            age_bin=("age_bin", "first"),
            is_healthy=("is_healthy", "first"),
            biological_sex=("biological_sex", "first"),
            model=("model", "first"),
        )
        .assign(
            aging_rate=lambda x: x.pred_age_delta / x.age_delta,
            gap_rate=lambda x: x.gap_delta / x.age_delta,
        )
        .reset_index()
    )
    pdf = pdf[pdf.age_delta >= min_years]
    pdf[gap_metric] = pdf.gap

    healthy_slope_pdf = (
        pdf[pdf.is_healthy & (pdf.age_delta >= min_years)]
        .groupby(["age_bin", "biological_sex"])
        .agg(
            healthy_gap=(gap_metric, "mean"),
            healthy_slope=("aging_rate", "mean"),
            n_healthy=(gap_metric, len),
        )
        .reset_index()
    )

    # merge healthy baselines in, compute adjusted gaps and slopes
    pdf = pdf.merge(
        healthy_slope_pdf, on=["age_bin", "biological_sex"], how="left"
    ).assign(
        slope_adj=lambda x: x.aging_rate - x.healthy_slope,
        slope_adj_days=lambda x: x.slope_adj * 365.25,
    )

    return pdf


def add_adjusted_age_gaps(preddf):
    models = preddf.model.unique()
    pdfs = []
    model_dicts = {}
    for m in models:
        print(f" ... adjusting model {m}")
        adf, mods = adjust_gap_by_age(preddf, model=m)
        model_dicts[m] = mods
        pdfs += [adf]
    return pd.concat(pdfs, axis=0), model_dicts


def adjust_gap_by_age(
    preddf, model="yhat-mod_global_weight_none-subset-all", splits=["train", "test"]
):
    ## subset to healthy/training subjects for specific model
    hdf = preddf[
        (preddf.model == model)
        & (preddf.is_healthy)
        & (preddf.split.isin(splits))
        & (preddf.age_bin != ">75")
    ]
    hdf = hdf[hdf.gap.notnull()]

    ## find healthy predictions and make adjustment such that
    ## E[ y - yhat | age, healthy] = 0.  Do so with a regression of the
    ## residual on age
    lin_adj_model = smf.ols("gap ~ age", data=hdf).fit()

    ## spline model
    bs = BSplines(
        hdf["age"],
        df=[8],
        degree=[3],
        knot_kwds=[
            {
                "upper_bound": preddf.age.max() + 0.1,
                "lower_bound": preddf.age.min() - 0.1,
            }
        ],
    )
    spline_adj_model = GLMGam.from_formula("gap ~ age", data=hdf, smoother=bs).fit()

    ## apply to all data in hdf
    pdf = preddf[(preddf.model == model)]
    pred_gap_lin = lin_adj_model.predict(pdf["age"])
    pred_gap_spline = spline_adj_model.predict(
        exog=pdf[["age"]],
        exog_smooth=pdf[["age"]],
    )
    pdf = pdf.assign(
        gap_adj_lin=lambda x: x.gap - pred_gap_lin,
        gap_adj_spline=lambda x: x.gap - pred_gap_spline,
    )
    return pdf, dict(adj_traindf=hdf[["gap", "age"]], bs=bs)


#############################
# Relative Risk Computation #
#############################


def add_per_group_quantiles(
    pdf,
    group_vars=["biological_sex", "age_bin"],
    health_metric="gap",
    no_grouping=False,
    n_groups=5,
    bins=None,
):
    """Creates a metric_group and metric_q column, that reflects, within
    the specified groupings, the quantile of the health_metric for each
    prediction (i.e., the 0th quantile of age gap)
    """

    if bins is None:
        cut_fn = lambda x: pd.qcut(x[health_metric], n_groups, duplicates="drop")
    else:
        cut_fn = lambda x: pd.cut(x[health_metric], bins)

    ## make metric quintile groups on the fly (e.g., for vo2max or adj_resid age gap)
    def per_group_q(gdf):
        # print(gdf.shape, condition, gdf.age_bin, gdf.sex)
        return gdf.assign(
            vo2max_group=lambda x: pd.qcut(x.vo2max_ave, n_groups),
            vo2max_q=lambda x: x.vo2max_group.cat.codes,
            metric_group=lambda x: cut_fn(x),
            metric_q=lambda x: x.metric_group.cat.codes,
            metric_name=health_metric,
        )

    pdf = pdf.groupby(group_vars, group_keys=False).apply(per_group_q)

    return pdf


def make_rel_risk_df(
    preddf,
    groups=["age_bin", "sex"],  # or age_bin, sex, vo2max_q
    condition="bloodpressure",
    health_metric="resid",  # vo2max_ave
    seed=0,
):
    """From model predictions, create a dataset of relative risks for condition rates
    Returns ratedf
        - age_bin
        - sex
        - model
        - q-group
        - variable (stat) (including n sub)
        - ci_str
    """
    pdf = preddf[
        preddf[condition].isin(["[no]", "[yes]"])
        & preddf.biological_sex.isin(["Male", "Female"])
    ]

    def compute_group_stats(gdf):
        # grab arrays
        is_yes = 1.0 * (gdf[condition].values == "[yes]")
        metric_q = gdf["metric_q"].values
        metric_group = gdf["metric_group"].values
        vo2max_group = gdf["vo2max_group"].values
        qs = np.sort(np.unique(metric_q))

        ## map metric_q to metric group name
        metric_q_name_map = (
            gdf[["metric_q", "metric_group"]]
            .drop_duplicates()
            .set_index("metric_q")["metric_group"]
            .to_dict()
        )
        metric_q_name_map[None] = "all"

        def _stats(is_yes, metric_q, bi):
            base_rate = np.mean(is_yes)
            if base_rate == 0.0:
                base_rate = 1e-6

            def _qres(q=None):
                ys = is_yes if q is None else is_yes[metric_q == q]
                qname = "group" if q is None else f"q{q}"
                if len(ys) == 0:
                    rate = 0.0
                else:
                    rate = np.mean(ys)
                return dict(
                    metric=health_metric,
                    metric_q=qname,
                    metric_group=str(metric_q_name_map[q]),
                    rate=rate,
                    base_rate=base_rate,
                    rel_risk=rate / base_rate,
                    abs_risk_inc=rate - base_rate,
                    rel_risk_inc=(rate - base_rate) / base_rate,
                    bidx=bi,
                    n_subj=len(ys),
                )

            res_list = [_qres()]
            for q in qs:
                res_list += [_qres(q)]
            return res_list

        rs = np.random.RandomState(seed)
        bs_list = _stats(is_yes, metric_q, bi=-1)
        for bi in range(1000):
            bidx = rs.choice(len(is_yes), size=len(is_yes))
            bs_list += [*_stats(is_yes[bidx], metric_q[bidx], bi)]
        bdf = pd.DataFrame(bs_list).melt(
            id_vars=["bidx", "metric_q", "metric_group", "metric"]
        )
        ## summarize by metric q and variable
        sumdf = (
            bdf.sort_values("bidx")
            .groupby(["metric_q", "variable"])
            .agg(
                metric=("metric", "first"),
                metric_group=("metric_group", "first"),
                ci_str=(
                    "value",
                    lambda x: conf_int_str(x.values[1:], xmid=x.values[0]),
                ),
                q_mid=("value", lambda x: x.values[0]),
                q_lo=("value", lambda x: np.percentile(x, 2.5)),
                q_hi=("value", lambda x: np.percentile(x, 97.5)),
                q_min=("value", "min"),
                q_max=("value", "max"),
            )
            # .reset_index()
            .assign(condition=condition)
        )
        return sumdf

    bootdf = pdf.groupby(groups).apply(compute_group_stats).reset_index()
    return bootdf


################
## Helpers    ##
################


def label_smoker(subjdf):
    ## if healthy (i.e., never smoker) label healthy
    ## otherwise, if "no" to smoke_100, label never smoker
    ## otherwise, label "has_smoked"
    subjdf = subjdf.assign(
        smoking_history=lambda x: np.where(
            x.is_healthy,
            "healthy",
            np.where(
                x.smoke_100 == "no",
                "never_smoker",
                np.where(x.smoke_100 == "yes", "has_smoked", None),
            ),
        ),
        smoker_type=lambda x: np.where(
            x.smoking_history == "has_smoked",
            x.smoke_current,
            "never_smoker",
        ),
        smoking_status=lambda x: np.where(
            x.smoking_history == "has_smoked", x.smoker_type, x.smoking_history
        ),
    )
    return subjdf


def conf_int_str(x, lo=2.5, hi=97.5, xmid=None):
    if xmid is None:
        val = np.round(np.median(x), 3)
    else:
        val = np.round(xmid, 3)
    lo, hi = np.percentile(x, [lo, hi])
    lo = np.round(lo, 3)
    hi = np.round(hi, 3)
    return f"{val:.3f} [{lo:.3f}-{hi:.3f}]"


def bootstrap_conf_int(x, estimator=np.mean, lo=2.5, hi=97.5, nboot=1000, seed=0):
    rs = np.random.RandomState(seed)
    xmid = estimator(x)
    n = len(x)
    bvals = []
    for _ in range(nboot):
        idx = rs.choice(n, size=n, replace=True)
        bvals += [estimator(x[idx])]
    ci_str = conf_int_str(bvals, lo=lo, hi=hi, xmid=xmid)
    blo, bhi = np.percentile(bvals, [lo, hi])
    return dict(
        bmid=xmid,
        blo=blo,
        bhi=bhi,
        b_ci_str=ci_str,
        biqr_lo=np.percentile(x, 25),
        biqr_hi=np.percentile(x, 75),
        biqr_mid=np.percentile(x, 50),
        biqr_ctr=conf_int_str(x, lo=25, hi=75, xmid=np.percentile(x, 50)),
        n=n,
    )


def bucket_age(ages):
    return pd.cut(
        ages,
        right=False,
        bins=[0, 25, 35, 45, 55, 65, 75, np.inf],
        labels=[
            "<25",
            "[25,35)",
            "[35,45)",
            "[45,55)",
            "[55,65)",
            "[65,75)",
            ">75",
        ],
    )
