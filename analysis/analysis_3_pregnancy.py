#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os
import statsmodels.formula.api as sm
from joblib import Parallel, delayed, parallel_backend
import pandas as pd
import numpy as np
import statsmodels.api as sm
from .viz import colors, plt, sns


def make_pregnancy_figure(preg_outcome_df, dailydf, subjdf, output_dir):
    ## Output for tables + figures
    OUTPUT_DIR = os.path.join(output_dir, "fig-pregnancy-analysis")
    os.makedirs(output_dir, exist_ok=True)

    ## Pregnancy Analysis --- 3 months (short)
    run_pregnancy_analysis(
        preg_outcome_df, dailydf, subjdf, mode="", output_dir=OUTPUT_DIR
    )


def run_pregnancy_analysis(preg_outcome_df, dailydf, subjdf, mode="", output_dir=""):
    """make plots for all pregnancy figures"""

    ## save in mode --- either short or long
    mode_outdir = os.path.join(output_dir, mode)
    os.makedirs(mode_outdir, exist_ok=True)
    mode_figname = lambda x: os.path.join(mode_outdir, x)

    ## Make pregnancy outcome dataframe --- one row per subject
    ## with first reported outcome date (either vaginal delivery or c section),
    ## subset to subjects with no pregnancy, at study start,
    subject_to_date_map = preg_outcome_df.set_index("canonical_subject_id")[
        "pregnancy_outcome_date"
    ].to_dict()
    print("No. of preg outcomes in survey data: ", len(subject_to_date_map))

    dailydf_subjs = dailydf.index.dropna().unique()
    subject_to_date_map = {
        k: v for k, v in subject_to_date_map.items() if k in dailydf_subjs
    }
    print(
        "No. of preg outcomes in survey data and PPG data: ", len(subject_to_date_map)
    )

    ## for each pregnant person, grab 9 month window before and look at slopes
    ## across all 3 month intervals
    statdf, seqdf = compute_period_statistics_dataframe(
        subject_to_date_map=subject_to_date_map,
        dailydf=dailydf,
        n_after=3,
    )
    statdf.to_csv(mode_figname("pregnancy-phase-ranges.csv"))
    print(
        "Number of subjects with proximal data: ", statdf.canonical_subject_id.nunique()
    )

    #########################################################################
    ## Analysis: subset to subjects with at least 20 observed days within
    ## each period, and report statistics
    #########################################################################
    periods_to_include = ["[-270, -180)", "[-180, -90)", "[-90, 0)", "[0, 90)"]
    if mode == "long":
        periods_to_include += ["[90, 180)", "[180, 270)", "[270, 360)"]
    min_days_per_period = 10
    cntdf = (
        statdf[statdf.period.isin(periods_to_include)]
        .groupby("canonical_subject_id")  # type: ignore
        .agg(
            min_nobs=("nobs", "min"),
            n_per=("period", len),
        )
        .reset_index()  # type: ignore
    )
    subj_to_keep = cntdf[
        (cntdf.min_nobs >= min_days_per_period)
        & (cntdf.n_per >= len(periods_to_include))
    ].canonical_subject_id.tolist()
    print(
        f"Filtering to {len(subj_to_keep)} subjects with at least "
        f"{min_days_per_period} daily age estimates per period"
    )
    subdf = statdf[
        statdf.canonical_subject_id.isin(subj_to_keep)
        & statdf.period.isin(periods_to_include)
    ]
    firstdf = subdf[subdf.period == periods_to_include[0]].rename(
        columns={"yhat_smooth_first": "first_period_yhat_smooth_first"}
    )
    subdf = (
        subdf.merge(
            firstdf[["canonical_subject_id", "first_period_yhat_smooth_first"]],
            on="canonical_subject_id",
            how="left",
        )
        .merge(preg_outcome_df, on="canonical_subject_id", how="left")
        .assign(
            slope_yr_per_yr=lambda x: x.slope * 365.25,
            slope_yr_per_90days=lambda x: x.slope * 90,
            pred_age_delta_from_start=lambda x: x.yhat_smooth_last
            - x.first_period_yhat_smooth_first,
            period_delta=lambda x: x.yhat_smooth_last - x.yhat_smooth_first,
        )
    )

    sub_seqdf = seqdf.loc[subj_to_keep]
    sub_seqdf = sub_seqdf[sub_seqdf.period.isin(periods_to_include)]
    print(subdf.shape)
    subdf = merge_subj_info(subdf, subjdf)

    ##
    ## make violin plot comparing slopes
    ##
    def make_boxplot(plot_type="box", slope_col="slope_yr_per_yr", hue=None):
        print(subdf.columns)
        plot_fn = sns.boxplot
        if plot_type == "violin":
            plot_fn = sns.violinplot
        elif plot_type == "point":
            plot_fn = sns.pointplot  # lambda **kwargs: sns.pointplot(**kwargs
        fig, ax = plt.figure(figsize=(4.5, 4.5)), plt.gca()
        plot_fn(
            x="period",
            y=slope_col,
            data=subdf,
            palette="dark:lightgrey",
            hue=hue,
            # dodge=False if hue is None else 0.1,
            # palette = sns.color_palette("colorblind")
        )
        unit = "Delta (years)"
        if slope_col == "slope_yr_per_yr":
            unit = "Slope (pred. years/cal. years)"
        elif slope_col in ["slope_yr_per_90days", "period_delta"]:
            unit = "Slope (pred. years/cal. 90 days)"
        ax.set_xlabel("Time period (days)")
        ax.set_ylabel(unit)
        fig.tight_layout()
        return fig

    plt.close("all")
    for slope_col in [
        "period_delta",
        "slope_yr_per_90days",
        "pred_age_delta_from_start",
    ]:
        for pt in ["box", "violin", "point"]:
            for hue in [
                None,
                "no_disease_history",
                "bloodpressure",
                "pregnancy_dm",
                "preeclampsia",
                "diabetes",
                "age_bin",
                "bmi_bin",
                "smoker_type",
            ]:
                print("slope, plot type, group:", slope_col, pt, hue)
                fig = make_boxplot(plot_type=pt, slope_col=slope_col, hue=hue)
                fig.savefig(
                    mode_figname(f"pregnancy-{slope_col}-{pt}-grp-{hue}.pdf"),
                    bbox_inches="tight",
                )

    ################################################################
    ## make pointplot comparison, hued by pregnancy outcome type  ##
    ################################################################
    plt.close("all")
    pdf_out = subdf
    for outcome in [
        "preeclampsia",
        "pregnancy_dm",
        "bloodpressure",
        "diabetes",
        "no_disease_history",
        "pregnancy_outcome_type",
    ]:
        fig, ax = plt.figure(figsize=(8, 4)), plt.gca()
        sns.pointplot(
            x="period",
            y="period_delta",
            hue=outcome,
            data=pdf_out,
            # dodge=0.2,
            ax=ax,
        )

    ## print statistics on increasers vs decreasers
    print(subdf.columns.tolist())
    mdf = subdf.melt(
        id_vars=["canonical_subject_id", "period", "pregnancy_outcome_type"],
        value_vars=[
            "period_delta",
            "slope_yr_per_90days",
            "pred_age_delta_from_start",
        ],
        value_name="metric_value",
    )
    sumdf = mdf.groupby(["period", "variable"]).agg(
        frac_above_0=("metric_value", lambda x: ci_str(x > 0)),
        frac_below_0=("metric_value", lambda x: ci_str(x < 0)),
        frac_above_90days=("metric_value", lambda x: ci_str(x > (365.25 / 90))),
    )
    sumdf.to_csv(mode_figname("delta-comparison-table.csv"))

    sdf = subdf.pivot(
        index="canonical_subject_id", columns="period", values="period_delta"
    ).reset_index()
    comps = [
        ("[-90, 0)", "[-180, -90)"),
        ("[-90, 0)", "[-270, -180)"),
        ("[-90, 0)", "[0, 90)"),
    ]
    print("Available periods to compare: ", sdf.head())
    compdf = pd.DataFrame(
        {f"period_delta-{a}>{b}": ci_str(sdf[a] > sdf[b]) for a, b in comps},
        index=["frac_greater"],
    ).T
    compdf.to_csv(mode_figname("delta-within-subj-comparison-table.csv"))

    ################################
    ## plot lowess statistics     ##
    ################################
    fig, ax = plt.figure(figsize=(8, 6)), plt.gca()
    cids = sub_seqdf.index.unique()
    tmin, tmax = sub_seqdf.days.min(), sub_seqdf.days.max()
    tgrid = np.arange(tmin, tmax)
    dfs = []
    for c in cids:
        ssdf = sub_seqdf.loc[c]
        smoothed = sm.nonparametric.lowess(
            exog=ssdf.days,
            endog=ssdf["yhat-mean"],
            frac=0.15,
            xvals=tgrid,
        )
        ## center on pred age in first 30 days
        initial_age = ssdf["yhat-mean"][ssdf.days < -240].mean()
        smoothed_centered = smoothed - initial_age
        print(initial_age)
        dfs += [
            pd.DataFrame(
                dict(
                    canonical_subject_id=c,
                    days=tgrid,
                    ysmooth=smoothed,
                    ysmooth_centered=smoothed_centered,
                )
            )
        ]
        ax.plot(tgrid, smoothed_centered, color="grey", alpha=0.2)

    smoothdf = (
        pd.concat(dfs, axis=0)
        .merge(
            subjdf[
                [
                    "canonical_subject_id",
                    "bmi",
                    "is_healthy",
                    "smoker_type",
                    "no_disease_history",
                    "bloodpressure",
                    "diabetes",
                    "age",
                ]
            ],
            on="canonical_subject_id",
            how="left",
        )
        .assign(
            bmi_bin=lambda x: pd.cut(x.bmi, [0, 23, 30, 100]),
            age_bin=lambda x: pd.cut(x.age, [0, 25, 35, 45]),
        )
        .merge(
            preg_outcome_df[
                [
                    "canonical_subject_id",
                    "pregnancy_outcome_type",
                    "pregnancy_dm",
                    "preeclampsia",
                ]
            ],
            how="left",
            on="canonical_subject_id",
        )
    )

    def plot_trends(
        group=None,
        levels=None,
        colors=colors,
    ):
        smdf = smoothdf
        group = "group" if group is None else group
        if group != "group":
            smdf = smdf[smdf[group] != "[do_not_know]"]
        ymu_grp = (
            smdf.assign(group="all")
            .groupby(["days", group])
            .agg(
                yc_mean=("ysmooth_centered", "mean"),
                yc_std=("ysmooth_centered", "std"),
                yc_q50=("ysmooth_centered", lambda x: np.nanpercentile(x, 50)),
                yc_q75=("ysmooth_centered", lambda x: np.nanpercentile(x, 75)),
                yc_q25=("ysmooth_centered", lambda x: np.nanpercentile(x, 25)),
                yc_cnt=("canonical_subject_id", lambda x: x.nunique()),
            )
            .reset_index()
        )
        levels = ymu_grp[group].unique()
        fig, ax = plt.figure(figsize=(7, 3)), plt.gca()
        for level, color in zip(levels, colors):
            ymu = ymu_grp[ymu_grp[group] == level]
            n = ymu.yc_cnt.unique()[0]
            ax.plot(
                ymu.days, ymu.yc_q50, color=color, label=f"{group} {str(level)} (n={n})"
            )
            ax.fill_between(
                ymu.days,
                ymu.yc_q25,
                ymu.yc_q75,
                color=color,
                alpha=0.5,
                # label=f"IQR, n={n}"
            )
            ax.set_xlabel("days before/after birth")
            ax.set_ylabel("pred. age above baseline")
            ax.set_xlim(-270, 89)
            if mode == "long":
                ax.set_xlim(-270, 359)
        fig.tight_layout()
        ax.legend(loc="upper left")
        return fig, ax, ymu_grp

    fig, ax, ymu_grp = plot_trends()
    fig.savefig(mode_figname("pregnancy-ts-summary.pdf"), bbox_inches="tight")
    ymu_grp.to_csv(mode_figname("pregnancy-ts-summary-data.csv"))

    for group in [
        "bloodpressure",
        "diabetes",
        "bmi_bin",
        "age_bin",
        "preeclampsia",
        "pregnancy_dm",
        "pregnancy_outcome_type",
    ]:
        fig, ax, ymu_grp = plot_trends(group=group)
        fig.savefig(
            mode_figname(f"pregnancy-ts-summary-{group}.pdf"), bbox_inches="tight"
        )
        ymu_grp.to_csv(mode_figname(f"pregnancy-ts-summary-data-{group}.csv"))

    ###############################################
    ## Plot daily time series for a few people   ##
    ###############################################
    cids_to_plot = (
        subdf[(subdf.period == "[-90, 0)")]
        .sort_values("period_delta")
        .canonical_subject_id.unique()
        .tolist()
    )
    ca, cb, cc = cids_to_plot[-2], cids_to_plot[-20], cids_to_plot[-56]
    fig, (axa, axb, axc) = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    fig, ax = plot_subject_periods(sub_seqdf.loc[ca], ax=axa)
    fig, ax = plot_subject_periods(sub_seqdf.loc[cb], ax=axb)
    fig, ax = plot_subject_periods(sub_seqdf.loc[cc], ax=axc)
    axc.legend(loc="lower right")
    axa.set_xlabel(None)
    axb.set_xlabel(None)
    fig.tight_layout()
    fig.savefig(mode_figname(f"pregnancy-ts-examples.pdf"), bbox_inches="tight")


def plot_subject_periods(subj_seqdf, ax=None, plot_slopes=False):
    if ax is None:
        _, ax = plt.figure(figsize=(10, 4)), plt.gca()
    ax.plot(
        subj_seqdf.days, subj_seqdf.yhat_smooth, c="black", label="$\hat{y}$ smoothed"
    )
    ylo, yhi = ax.get_ylim()
    ylo = ylo - 3
    yhi = yhi + 3
    ax.scatter(
        subj_seqdf.days, subj_seqdf["yhat-mean"], s=2, c="black", label="$\hat{y}$"
    )
    ax.set_ylim(ylo, yhi)
    colors = sns.color_palette("colorblind")[1:]
    pers = subj_seqdf.period.dropna().unique()
    xlo, xhi = None, None
    for i, period in enumerate(pers):
        print(period)
        pdf = subj_seqdf[subj_seqdf.period == period]
        ds, de = period[1:-1].split(",")
        if plot_slopes:
            ax.plot(pdf.days, pdf.period_yhat, color=colors[i])
        ax.axvspan(int(ds), int(de), facecolor=colors[i], alpha=0.2, zorder=-1)
        if i == 0:
            xlo = int(ds)
        if i == (len(pers) - 1):
            xhi = int(de)
    ax.vlines(
        0,
        ymin=ylo,
        ymax=yhi,
        color="darkred",
        linestyle="--",
        linewidth=3,
        label="outcome date",
    )
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_ylabel("Estimated Age")
    ax.set_xlabel("Days before/after outcome")
    return ax.figure, ax


def ci_str(x, fun=np.mean, n_boot=1000):
    rs = np.random.RandomState(0)
    bs = []
    for _ in range(n_boot):
        idx = rs.choice(len(x.values), size=len(x.values))
        bs += [fun(x.values[idx])]
    ## total statistics
    b_total = np.round(fun(x.values), 2)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    lo, hi = np.round(lo, 2), np.round(hi, 2)
    return f"{b_total:.2f} [{lo:.2f} - {hi:.2f}]"


def merge_subj_info(df, subjdf):
    print(df.columns)
    return (
        df.drop(
            columns=[
                "bmi",
                "is_healthy",
                "smoker_type",
                "no_disease_history",
                "bloodpressure",
                "diabetes",
                "age",
            ]
        )
        .merge(
            subjdf[
                [
                    "canonical_subject_id",
                    "bmi",
                    "is_healthy",
                    "smoker_type",
                    "no_disease_history",
                    "bloodpressure",
                    "diabetes",
                    "age",
                ]
            ],
            on="canonical_subject_id",
            how="left",
        )
        .assign(
            bmi_bin=lambda x: pd.cut(x.bmi, [0, 23, 30, 100]),
            age_bin=lambda x: pd.cut(x.age, [0, 25, 35, 45]),
        )
    )


def summarize_subject_time_periods(
    subj_dailydf,
    outcome_date,
    time_delta="90 days",
    n_before=3,
    n_after=2,
    lowess_frac=0.15,
):
    """Summarize the slopes for three month periods before and after
    pregnancy outcome date.

    Args:
        - subj_dailydf: dataframe with columns
            `local_date`
            `yhat-mean`: age prediction (averaged over all daily PPGs)
            `n_ppg`: number of ppgs averaged that day
        - outcome_date: date of pregnancy outcome considered for this one
            individual.

    Returns:
        - slopedf: dataframe with columns
            `period`: "first", "mid", "last", "post",
            `delta`: overal range of yhat within period
            `subject_sd`: overall standard deviation of subject within period
            `slope`: slope over 3 month periods
            `intercept`: intercept over 3 month period
            `nobs`: number of days with observations within 3 month period
        - yhat_df: daily predictions, but with labeled periods and linear
            within period predictions (for plotting)

    """
    n_unique_subj = subj_dailydf.index.nunique()
    assert (
        n_unique_subj == 1
    ), f"found {n_unique_subj}, expected 1, {str(subj_dailydf.index.unique())}"
    cid = subj_dailydf.index[0]
    dt = pd.Timedelta(time_delta)

    subj_dailydf = subj_dailydf.sort_values("local_date").assign(
        days=lambda x: (x.local_date - outcome_date).dt.days,
    )
    smoothed = sm.nonparametric.lowess(
        exog=subj_dailydf.days,
        endog=subj_dailydf["yhat-mean"],
        frac=lowess_frac,
    )
    subj_dailydf = subj_dailydf.assign(yhat_smooth=smoothed[:, 1])

    def summarize_(df, period):
        if df.shape[0] == 0:
            return None, pd.DataFrame(dict(local_date=None, nobs=0), index=[""])
        delta = df.yhat_smooth.max() - df.yhat_smooth.min()
        daygrid = (df.local_date - df.local_date.min()).dt.days.values
        X = sm.add_constant(daygrid)
        beta, *_ = np.linalg.lstsq(X, df["yhat-mean"].values, rcond=None)
        intercept, slope = beta
        slope_dict = dict(
            delta=delta,
            period=period,
            subject_sd=ddf.yhat_smooth.std(),
            canonical_subject_id=cid,
            slope=slope,
            intercept=intercept,
            nobs=df.shape[0],
            yhat_max=df["yhat-mean"].max(),
            yhat_min=df["yhat-mean"].min(),
            yhat_smooth_max=df["yhat_smooth"].max(),
            yhat_smooth_min=df["yhat_smooth"].min(),
            yhat_mean=df["yhat-mean"].mean(),
            yhat_smooth_mean=df["yhat_smooth"].mean(),
            yhat_smooth_first=df["yhat_smooth"].iloc[0],
            yhat_smooth_last=df["yhat_smooth"].iloc[-1],
            yhat_smooth_delta=df["yhat_smooth"].iloc[-1] - df["yhat_smooth"].iloc[0],
        )
        yhat_df = df.assign(period_yhat=np.dot(X, beta), period=period)
        return slope_dict, yhat_df

    ## go through 90 day periods
    period_list = []
    for n_periods_before in np.arange(-n_before, n_after):
        period_start = outcome_date + n_periods_before * dt
        period_end = period_start + dt
        start_days = (n_periods_before * dt).days
        period_name = f"[{start_days}, {start_days+dt.days})"
        ddf = subj_dailydf[
            (subj_dailydf.local_date >= period_start)
            & (subj_dailydf.local_date < period_end)
        ]
        period_list += [summarize_(ddf, period_name)]

    slopedf = pd.DataFrame([a for a, _ in period_list if a is not None])
    subj_ddf = pd.concat([b for _, b in period_list if b is not None]).sort_values(
        "local_date"
    )
    return slopedf, subj_ddf


def summarize_slope_change(subj_dailydf, outcome_date, time_delta="180 days"):
    n_unique_subj = subj_dailydf.index.nunique()
    assert (
        n_unique_subj == 1
    ), f"found {n_unique_subj}, expected 1, {str(subj_dailydf.index.unique())}"
    cid = subj_dailydf.index[0]
    dt = pd.Timedelta(time_delta)

    subj_dailydf = subj_dailydf.sort_values("local_date").assign(
        days=lambda x: (x.local_date - outcome_date).dt.days,
        yrs=lambda x: x.days / 365.25,
        y=lambda x: x["yhat-mean"],
        post_treat=lambda x: 1.0 * (x.local_date > outcome_date),
        post_treat_yrs=lambda x: x.post_treat * x.yrs,
        post_treat_days=lambda x: x.post_treat * x.days,
        period=lambda x: np.where(x.post_treat, "post", "pre"),
    )
    subj_dailydf = subj_dailydf[
        (subj_dailydf.local_date > (outcome_date - dt))
        & (subj_dailydf.local_date < (outcome_date + dt))
    ]
    nobs_post = int(subj_dailydf.post_treat.sum())
    nobs_pre = int((1 - subj_dailydf.post_treat).sum())
    if (nobs_post < 10) | (nobs_pre < 10):
        return None, None

    res = sm.ols(formula="y ~ days + post_treat_days", data=subj_dailydf).fit()
    resdf = (
        pd.concat(
            [
                res.params.rename("coef"),
                res.conf_int(alpha=0.05).rename(columns={0: "q2.5", 1: "q97.5"}),
                res.pvalues.rename("pvalue"),
            ],
            axis=1,
        )
        .reset_index()
        .rename(columns={"index": "variable"})
        .assign(
            cid=cid,
            time_delta=time_delta,
            nobs_post=nobs_post,
            nobs_pre=nobs_pre,
        )
    )
    subj_dailydf = subj_dailydf.assign(slope_model_yhat=res.predict())
    return resdf, subj_dailydf


def compute_slope_change_models(subject_to_date_map, dailydf, **kwargs):
    ## unpack canonical subject ids and corresponding dates
    cids = [c for c in subject_to_date_map.keys()]
    outcome_dates = [subject_to_date_map[c] for c in cids]
    ## unpack daily dataframes (might make joblib call faster)
    sub_dailydf = dailydf.loc[cids]
    ddfs = [sub_dailydf.loc[[cid]].copy() for cid in cids]
    with parallel_backend("loky", inner_max_num_threads=40):
        rows = Parallel(n_jobs=40, verbose=20)(
            delayed(summarize_slope_change)(df, od, **kwargs)
            for df, od in zip(ddfs, outcome_dates)
        )
    statdf = pd.concat([r for r, _ in rows], axis=0)
    seqdf = pd.concat([s for _, s in rows], axis=0)
    return statdf, seqdf


def compute_period_statistics_dataframe(subject_to_date_map, dailydf, **kwargs):
    """For each subject in the preg_outcome_df, compute statistics about
    3 month (90 day) periods immediately preceding and following
    (i.e., 270-180, 180-90, 90-0 before and 0-90 days after) a pregnancy outcome
    of either a c section or vaginal birth.  Compute summary statistics for
    each of these periods relating to predicted age.
    """

    ## unpack canonical subject ids and corresponding dates
    cids = [c for c in subject_to_date_map.keys()]
    outcome_dates = [subject_to_date_map[c] for c in cids]

    ## unpack daily dataframes (might make joblib call faster)
    dailydf = dailydf.set_index("canonical_subject_id")
    ddfs = [dailydf.loc[cid].copy() for cid in cids]
    with parallel_backend("loky", inner_max_num_threads=40):
        rows = Parallel(n_jobs=40, verbose=20)(
            delayed(summarize_subject_time_periods)(df, od, **kwargs)
            for df, od in zip(ddfs, outcome_dates)
        )
    statdf = pd.concat([r for r, _ in rows], axis=0)
    seqdf = pd.concat([s for _, s in rows], axis=0)
    return statdf, seqdf
