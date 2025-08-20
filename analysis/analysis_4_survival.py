#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter, KaplanMeierFitter

FIGSIZE = (10, 4)
SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 22
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def fit_surv_model(
    survdf,
    form,
    outcome_col="is_event",
    outcome_date_col="end_date",
    penalty=0.0001,
):
    """
    survdf: survival analysis df created in make_survival_outcome_df
    form: model formula for regression
    outcome_col: name of outcome column for event vs non-event (default: is_event)
    outcome_date_col: name of date column with event dates (years to event)
       for events, and censor date (years to censoring) for non-events
    penalty: L2 penalty on Cox regression, sometimes needed for stability
        if have lots of covariates and relatively few events
    """

    ## can also add other preprocessing as desired here, eg to subset to
    ## cohorts without a given condition at baseline

    ## add a small L2 penalty for numeric stability
    mod = CoxPHFitter(penalizer=penalty, l1_ratio=0)

    mod.fit(
        survdf,
        duration_col=outcome_date_col,
        event_col=outcome_col,
        formula=form,
    )
    mod.print_summary(decimals=4)
    return mod


def fit_km_model(
    survdf,
    subgroup_name,
    outcome_col="is_event",
    outcome_date_col="end_date",
    table_times=[0, 1, 2, 3, 4],
):
    """
    Fit KM estimate within subgroups of age gap, then return
    the model, a plotting df, and a df to create an at-risk table
    on the plot we'll later create
    """

    ## can also add other preprocessing as desired here, eg to subset to
    ## cohorts without a given condition at baseline

    mod = KaplanMeierFitter()
    mod.fit(
        durations=survdf[outcome_date_col],
        event_observed=survdf[outcome_col],
        label=subgroup_name,
    )

    ## get stuff we need for plotting, since it needs to be custom
    ## surv curve + CIs
    plotdf = mod.survival_function_.reset_index()
    plotdf = plotdf.merge(
        mod.confidence_interval_survival_function_.reset_index().rename(
            columns={"index": "timeline"}
        )
    )

    table = mod.event_table.reset_index()
    at_risk_table = []
    for t in table_times:
        # first time at/after each table time
        ind = table.loc[lambda x: x.event_at >= t].index[0]
        at_risk = int(table.iloc[ind].at_risk)
        n_c = table.iloc[:ind].censored.sum()
        n_e = table.iloc[:ind].observed.sum()
        at_risk_table.append(
            {"time": t, "at_risk": at_risk, "censored": n_c, "events": n_e}
        )
    at_risk_table = pd.DataFrame(at_risk_table)

    return (mod, plotdf, at_risk_table)


def get_model_table(mod):
    """
    Helper function to get some relevant info from fitted Cox model
    mod as input, and write out a pd df with some model info
    """
    mod_table = pd.DataFrame(
        {
            "covariate": mod.params_.index.values,
            "covariate mean": mod._norm_mean.values,
            "covariate std": mod._norm_std.values,
            "coef": mod.params_.values,
            "hazard ratio": mod.hazard_ratios_.values,
            "coef_CI_lo": mod.confidence_intervals_.values[:, 0],
            "coef_CI_hi": mod.confidence_intervals_.values[:, 1],
            "p value": mod._compute_p_values(),
        }
    ).round(
        {
            "p value": 3,
            "covariate mean": 2,
            "covariate std": 1,
            "coef": 3,
            "hazard ratio": 3,
            "coef_CI_lo": 2,
            "coef_CI_hi": 2,
        }
    )
    mod_table["p value"] = mod_table["p value"].astype(str)
    mod_table.loc[lambda x: x["p value"] == "0.0", ["p value"]] = "<.001"
    return mod_table


def run_KM_analysis_and_plot(survdf, output_dir):
    ##
    ## setup for KM curves by age gap subgroup
    ##
    df0 = survdf.loc[lambda x: x.gap_adj_spline <= -1]
    df1 = survdf.loc[lambda x: (x.gap_adj_spline > -1) & (x.gap_adj_spline <= 1)]
    df2 = survdf.loc[lambda x: (x.gap_adj_spline > 1)]
    dfs = [df0, df1, df2]
    colors = ["green", "orange", "red"]
    table_times = [0, 1, 2, 3, 4]
    n_es = [int(_df.is_event.sum()) for _df in dfs]
    labels = [
        f"age gap <= -1 ({n_es[0]} events)",
        f"0 < age gap <= 1 ({n_es[1]} events)",
        f"age gap > 1 ({n_es[2]} events)",
    ]

    ### KM plots by age gap subgroup, with CIs
    FIGSIZE = (12, 8)
    SMALL_SIZE = 13
    plt.close("all")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, (_df, label, color) in enumerate(zip(dfs, labels, colors)):

        (mod, plotdf, at_risk_table) = fit_km_model(
            _df,
            subgroup_name=label,
            table_times=table_times,
        )
        plt.plot(plotdf["timeline"], plotdf[label], color=color, label=label)
        plt.fill_between(
            plotdf["timeline"],
            plotdf[f"{label}_lower_0.95"],
            plotdf[f"{label}_upper_0.95"],
            color=color,
            alpha=0.3,
        )

        ### add text for the at-risk table to the plot
        ### don't use ax.table, instead manually spell out all the text we need
        text_x_start = -1
        text_y_start = -0.3
        group_y_gap = 0.35
        y_gap = 0.07
        x_gap = 0.05
        plt.text(
            x=text_x_start,
            y=text_y_start - group_y_gap * i,
            s=label.split("(")[0],
            transform=ax.get_xaxis_transform(),
            fontsize=SMALL_SIZE,
        )
        plt.text(
            x=text_x_start,
            y=text_y_start - y_gap - group_y_gap * i,
            s="at risk",
            transform=ax.get_xaxis_transform(),
            fontsize=SMALL_SIZE,
        )
        for t, v in zip(table_times, at_risk_table.at_risk.values):
            plt.text(
                x=t - x_gap,
                y=text_y_start - y_gap - group_y_gap * i,
                s=v,
                transform=ax.get_xaxis_transform(),
                fontsize=SMALL_SIZE,
            )
        plt.text(
            x=text_x_start,
            y=text_y_start - y_gap * 2 - group_y_gap * i,
            s="censored",
            transform=ax.get_xaxis_transform(),
            fontsize=SMALL_SIZE,
        )
        for t, v in zip(table_times, at_risk_table.censored.values):
            plt.text(
                x=t - x_gap,
                y=text_y_start - y_gap * 2 - group_y_gap * i,
                s=v,
                transform=ax.get_xaxis_transform(),
                fontsize=SMALL_SIZE,
            )
        plt.text(
            x=text_x_start,
            y=text_y_start - y_gap * 3 - group_y_gap * i,
            s="events",
            transform=ax.get_xaxis_transform(),
            fontsize=SMALL_SIZE,
        )
        for t, v in zip(table_times, at_risk_table.events.values):
            plt.text(
                x=t - x_gap,
                y=text_y_start - y_gap * 3 - group_y_gap * i,
                s=v,
                transform=ax.get_xaxis_transform(),
                fontsize=SMALL_SIZE,
            )
        ## loop through and plot text for the at risk, censored, events

    plt.subplots_adjust(bottom=0.5, left=0.15)
    plt.grid(alpha=0.3)
    plt.title("KM curves, by age gap subgroup")
    plt.legend()
    plt.ylabel("Survival probability")
    plt.xlabel("Years")
    plt.savefig(
        os.path.join(output_dir, "surv_model_KM_curves_agegap.pdf"), bbox_inches="tight"
    )


def get_cov_means_by_age(survdf, ages):
    """
    When plotting fitted survival curves for different ages,
      if we vary only age but keep all other covariates
      (e.g. baseline medical conditions, Vo2max, etc) the same,
      we're ignorning the fact that these covariates all tend to change with age.
      So we'll get the average values of all our other covariates, for each age group.
    """
    sexes = []
    bps = []
    chols = []
    dms = []
    s_everys = []
    s_somes = []
    s_pasts = []
    bmis = []
    vo2s = []
    # get average covariates for age buckets defined by +/- 5 years
    # per age in ages input
    for age in ages:
        df = survdf.loc[lambda x: (x.age >= age - 5) & (x.age <= age + 5)]
        sexes.append(df.sex_is_female.mean())
        bps.append(df.hx_bp.mean())
        dms.append(df.hx_dm.mean())
        chols.append(df.hx_chol.mean())
        s_everys.append(df.smoke_everyday.mean())
        s_somes.append(df.smoke_somedays.mean())
        s_pasts.append(df.smoke_past.mean())
        bmis.append(df.bmi.mean())
        vo2s.append(df.vo2max_ave.mean())
    plot_cov_vals = np.concatenate(
        [
            np.array(ages)[:, None],
            np.array(sexes)[:, None],
            np.array(bps)[:, None],
            np.array(chols)[:, None],
            np.array(dms)[:, None],
            np.array(s_everys)[:, None],
            np.array(s_somes)[:, None],
            np.array(s_pasts)[:, None],
            np.array(bmis)[:, None],
            np.array(vo2s)[:, None],
        ],
        axis=1,
    )
    return plot_cov_vals


def run_cox_model_and_plot(survdf, output_dir):
    """
    Main function for fitting Cox model, saving out results table,
    and saving out fitted survival plots by age and age+age_gap subgroups
    """
    cox_form = "sex_is_female + age + gap_adj_spline + hx_bp + \
    hx_chol + hx_dm + smoke_everyday + smoke_somedays + smoke_past + \
    bmi + vo2max_ave"

    ### fit Cox model
    mod = fit_surv_model(survdf, form=cox_form)
    mod_table = get_model_table(mod)
    print(mod_table)
    mod_table.to_csv(os.path.join(output_dir, "surv_cox_model_table.csv"))

    FIGSIZE = (10, 4)
    SMALL_SIZE = 14
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 22

    ##
    ## Fitted survival curves for a few different ages
    ##
    ages = [35, 45, 55, 65, 75]
    plt.close("all")
    ax = mod.plot_partial_effects_on_outcome(
        covariates=[
            "age",
            "sex_is_female",
            "hx_bp",
            "hx_chol",
            "hx_dm",
            "smoke_everyday",
            "smoke_somedays",
            "smoke_past",
            "bmi",
            "vo2max_ave",
        ],
        values=get_cov_means_by_age(survdf, ages),
        cmap="coolwarm",
        plot_baseline=False,
        grid=True,
        figsize=FIGSIZE,
    )
    ax.legend(labels=[f"age={x}" for x in ages])
    ax.set_xlabel("Years")
    ax.set_ylabel("Survival probability")
    ax.set_title("Survival curves by age")
    ax.set_xlim([0, 5.0])
    plt.savefig(
        os.path.join(output_dir, "cox_model_surv_curves_age.pdf"), bbox_inches="tight"
    )

    ##
    ## Fitted survival curves for 2 ages and a few different age gaps
    ##
    ages = [55, 65]
    age_gaps = [-6, 0, 6]
    num_age_gaps = len(age_gaps)
    plot_cov_vals = get_cov_means_by_age(survdf, ages)
    ## again get average covariate values per age bin, and add on a few
    ##   age gaps to show what happens when we intervene on age gap all else fixed
    plot_cov_vals = np.concatenate(
        [
            np.concatenate(
                [
                    np.tile(plot_cov_vals[0, :], (len(age_gaps), 1)),
                    np.array(age_gaps)[:, None],
                ],
                axis=1,
            ),
            np.concatenate(
                [
                    np.tile(plot_cov_vals[1, :], (len(age_gaps), 1)),
                    np.array(age_gaps)[:, None],
                ],
                axis=1,
            ),
        ],
        axis=0,
    )

    plt.close("all")
    ax = mod.plot_partial_effects_on_outcome(
        covariates=[
            "age",
            "sex_is_female",
            "hx_bp",
            "hx_chol",
            "hx_dm",
            "smoke_everyday",
            "smoke_somedays",
            "smoke_past",
            "bmi",
            "vo2max_ave",
            "gap_adj_spline",
        ],
        values=plot_cov_vals[:num_age_gaps,],
        cmap="coolwarm",
        plot_baseline=False,
        grid=True,
        figsize=FIGSIZE,
    )
    ax = mod.plot_partial_effects_on_outcome(
        covariates=[
            "age",
            "sex_is_female",
            "hx_bp",
            "hx_chol",
            "hx_dm",
            "smoke_everyday",
            "smoke_somedays",
            "smoke_past",
            "bmi",
            "vo2max_ave",
            "gap_adj_spline",
        ],
        values=plot_cov_vals[num_age_gaps:,],
        cmap="coolwarm",
        plot_baseline=False,
        ax=ax,
        grid=True,
        ls="--",
    )
    ax.legend(labels=[f"age={x}, age gap={y}" for x in ages for y in age_gaps], ncols=2)
    ax.set_title("Survival curves by age and age gap")
    ax.set_xlabel("Years")
    ax.set_ylabel("Survival probability")
    ax.set_xlim([0, 5.0])
    plt.savefig(
        os.path.join(output_dir, "cox_model_surv_curves_age_agegap.pdf"),
        bbox_inches="tight",
    )


def make_survival_figures(survdf, output_dir):
    """
    Main function to run all survival analyses and create plots
    and results tables to write out to output_dir
    """
    run_KM_analysis_and_plot(survdf, output_dir)
    run_cox_model_and_plot(survdf, output_dir)
