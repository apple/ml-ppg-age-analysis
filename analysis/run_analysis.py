#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import os, logging, argparse
import pandas as pd
import analysis.risk_util as ru
from analysis.analysis_1_cohort import make_cohort_figure
from analysis.analysis_2_errors import make_error_figure
from analysis.analysis_3_pregnancy import make_pregnancy_figure
from analysis.analysis_4_survival import make_survival_figures
from analysis.analysis_supplemental_residuals import make_supp_residual_fig
from analysis.analysis_supplemental_sex_disparities import make_sex_disparity_fig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

#########
## CLI ##
#########
parser = argparse.ArgumentParser()
parser.add_argument(
    "-a",
    "--artifact-dir",
    type=str,
    default="./artifacts/",
)
args = parser.parse_args()


## paths/io
OUTPUT_DIR = os.path.join(args.artifact_dir, "example-output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = lambda name: os.path.join(OUTPUT_DIR, name)


def load_data(name):
    dpath = lambda name: os.path.join(args.artifact_dir, "age-model-data", name)
    return pd.read_csv(dpath(name))


def main():

    ### load age prediction dataframes + subject info
    # yhatdf = load_data("segment-df.csv")
    preddf = load_data("month-df.csv")
    subjdf = load_data("subj-df.csv")

    ## save plotting data
    make_plotting_data(preddf, subjdf)

    ### make cohort figures
    make_cohort_figure(preddf, subjdf, output_dir=OUTPUT_DIR)

    ### error figure
    make_error_figure(preddf, output_dir=OUTPUT_DIR)

    ## make pregnancy figure
    dailydf = load_data("daily-df.csv").assign(
        local_date=lambda x: pd.to_datetime(x.local_date)
    )
    preg_outcome_df = load_data("pregnancy-df.csv").assign(
        pregnancy_outcome_date=lambda x: pd.to_datetime(x.pregnancy_outcome_date)
    )
    make_pregnancy_figure(preg_outcome_df, dailydf, subjdf, output_dir=OUTPUT_DIR)

    ## supplemental bland altman
    make_supp_residual_fig(preddf, output_dir=OUTPUT_DIR)

    ## sex disparity fig
    make_sex_disparity_fig(preddf, subjdf, output_dir=OUTPUT_DIR)

    ## run survival analyses and make plots 
    survdf = load_data("survival-df.csv")
    make_survival_figures(survdf, output_dir=OUTPUT_DIR)


def make_plotting_data(preddf, subjdf):
    ## Make relrisk dataframes + save (plot with R)
    rrdf, rrdf_vo2max, bindf = ru.make_age_gap_medhistdf(preddf)
    rrdf.to_csv(out_path("medhist-ratedf.csv"))
    rrdf_vo2max.to_csv(out_path("medhist-vo2-ratedf.csv"))
    bindf.to_csv(out_path("medhist-ratedf-bins.csv"))

    rrdf, rrdf_vo2max, bindf = ru.make_age_gap_medhistdf(
        preddf, fix_age_gap_buckets=True
    )
    rrdf.to_csv(out_path("medhist-ratedf-fixed-gaps.csv"))
    rrdf_vo2max.to_csv(out_path("medhist-vo2-ratedf-fixed-gaps.csv"))
    bindf.to_csv(out_path("medhist-ratedf-fixed-gap-bins.csv"))

    ## Look at slopes among those > 1 year
    rrdf = ru.make_age_rate_medhistdf(preddf=preddf, subjdf=subjdf)
    rrdf.to_csv(out_path("medhist-slope-ratedf.csv"))

    ## Years / Slope added above healthy by smoking status
    (
        years_added_df,
        years_added_df_adj,
    ) = ru.make_years_above_healthy_df(preddf, subjdf, outcome="smoking_status")
    years_added_df.to_csv(out_path("years-added-smoking-df.csv"))
    years_added_df_adj.to_csv(out_path("years-added-smoking-df-adj.csv"))


if __name__ == "__main__":
    main()
