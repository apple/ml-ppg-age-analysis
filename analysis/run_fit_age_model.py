#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import os, logging, argparse, re, ast, pickle
import pandas as pd
import numpy as np
import analysis.analysis_io as io
from sklearn.linear_model import RidgeCV

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

parser = argparse.ArgumentParser()
parser.add_argument(
    "-e",
    "--embeddings",
    type=str,
    default="./artifacts/test_inferences.csv",
)
args = parser.parse_args()


def main():

    ## load test inferences --- in the paper, we aggregate into first subject month
    embeddf = pd.read_csv(args.embeddings, header=0)

    ## example: add synthetic ages and is_healthy designation --- ideally these would
    ## already be in the embeddf
    embeddf = embeddf.assign(
        embeddings=lambda x: x.embeddings.apply(convert_to_list),
        canonical_subject_id=lambda x: np.arange(x.shape[0]),
        approx_age=lambda x: np.random.uniform(low=18, high=90, size=x.shape[0]),
        n_seg=lambda x: [30] * x.shape[0],
    )
    subjdf = pd.DataFrame(
        dict(
            canonical_subject_id=np.arange(embeddf.shape[0]),
            is_healthy=np.random.choice([False, True], size=embeddf.shape[0]),
        )
    )

    ##
    ## Fit models for healthy subjects, save
    ##
    resdict = fit_subject_model(embeddf, subjdf)
    output_dir = os.path.dirname(args.embeddings)
    with open(os.path.join(output_dir, "age-model.pkl"), "wb") as f:
        pickle.dump(resdict, f)

    ##
    ## apply model to all embeddings
    ##
    embeddf = embeddf.assign(
        yhats=lambda x: x.embeddings.apply(
            lambda xx: resdict["mod"].predict(np.atleast_2d(xx)).squeeze()
        )
    )

    ###
    ### Switch here --- save synthetic ages for analysis tasks
    ###

    ## load embedding dataframe + subject side information
    ### load age prediction dataframes + subject info
    yhatdf = io.load_segment_predictions()
    preddf = io.make_subject_month_predictions(yhatdf)
    subjdf = preddf.drop_duplicates("canonical_subject_id", keep="first")
    dailydf = io.make_dailydf(yhatdf)
    preg_outcome_df = io.make_pregnancy_outcome_df(subjdf)
    survdf = io.make_survival_outcome_df(subjdf)

    ## save out all predictions into artifacts
    out_dir = os.path.join(output_dir, "age-model-data")
    os.makedirs(out_dir, exist_ok=True)
    yhatdf.to_csv(os.path.join(out_dir, "segment-df.csv"))
    preddf.to_csv(os.path.join(out_dir, "month-df.csv"))
    subjdf.to_csv(os.path.join(out_dir, "subj-df.csv"))
    dailydf.to_csv(os.path.join(out_dir, "daily-df.csv"))
    preg_outcome_df.to_csv(os.path.join(out_dir, "pregnancy-df.csv"))
    survdf.to_csv(os.path.join(out_dir, "survival-df.csv"))

    return


def fit_subject_model(embeddf, subjdf, train_healthy=True, seed=0, frac_train=0.75):
    """Trains only on a fraction of healthy subjects.
    Returns dictionary with healthy model info
    """

    ## label train/test split based on health status
    subjdf = label_split(
        subjdf, train_healthy=train_healthy, seed=seed, frac_train=frac_train
    )

    ## Restrict to ages in 18 to 90
    embeddf = embeddf[
        embeddf.approx_age.notnull()
        & (embeddf.approx_age >= 18.0)
        & (embeddf.approx_age <= 90.0)
        & (embeddf.n_seg >= 30)
    ]

    datadf = embeddf.merge(
        subjdf,  # [static_cols],
        how="left",
        on="canonical_subject_id",
    )

    ## unpack data
    X = np.vstack(datadf.embeddings.tolist())
    print(X)
    print(np.isnan(X).sum())
    y = datadf[["approx_age"]].values
    # determine training set
    train_idx = (datadf.split == "train").values

    # fit with ridge
    mod = RidgeCV(alphas=[1e-10, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]).fit(
        X=X[train_idx, :],
        y=y[train_idx, :].squeeze(),
    )

    # look at error statistics
    yhat = mod.predict(X)
    preddf = pd.DataFrame(
        dict(
            yhat=yhat,
            ytrue=y.squeeze(),
            resid=y.squeeze() - yhat,
            is_healthy=datadf.is_healthy.values,
            split=datadf.split.values,
            canonical_subject_id=datadf.canonical_subject_id.values,
        )
    )
    errdf = preddf.groupby(["split", "is_healthy"]).agg(
        mae=("resid", lambda x: np.mean(np.abs(x))),
        n_subj=("canonical_subject_id", lambda x: x.nunique()),
    )
    logging.info(errdf)

    ## save training subject IDs
    train_subj = datadf[train_idx].canonical_subject_id.unique().tolist()
    return dict(mod=mod, errdf=errdf, train_subj=train_subj)


def label_split(subjdf, train_healthy=True, seed=0, frac_train=0.75):
    rs = np.random.RandomState(seed)
    ## label split --- save split subjects
    healthy_subj = subjdf[subjdf.is_healthy].canonical_subject_id.tolist()
    if not train_healthy:
        healthy_subj = subjdf.canonical_subject_id.tolist()[: len(healthy_subj)]
    logging.info(f"Train subj {len(healthy_subj)}, train healthy = {train_healthy}")
    subj_perm = rs.permutation(sorted(healthy_subj))
    train_subj = subj_perm[: int(frac_train * len(healthy_subj))]
    train_dict = {sid: "train" for sid in train_subj}
    label_ = lambda sid: "train" if sid in train_dict else "test"
    return subjdf.assign(
        split=lambda x: x.canonical_subject_id.apply(label_),
    )


def label_subject_health(subjdf):
    """Expects pandas loaded dataframe"""
    return subjdf.assign(
        no_meds=lambda x: x.medication_q1 == "[no]",
        never_smoke=lambda x: x.smoke_100 == "no",
        no_disease_history=lambda x: has_no_disease_history(x),
        is_healthy=lambda x: (x.no_meds) & (x.never_smoke) & (x.no_disease_history),
    )


def has_no_disease_history(subjdf):
    """Returns boolean array, true indicates a no response to each of the medhist
    column questions"""
    ## medical history -- subjects must report "no" to all of these
    ## categories to be considered "healthy"
    medhist_cols = [
        # metabolic
        "afib",
        "heartrhythm",
        "heartdisease",
        "diabetes",
        "bloodpressure",
        "cholesterol",
        "heartattack",
        "heartfailure",
        "pacemaker",
        "stroke_or_tia",
        "arterydisease",
        # bone/muscle/joint/mobility
        "arthritis",
        "hipknee",
        "lowback",
        "neckdisorder",
        "osteoporosis",
        # other conditions
        "asthma",
        "sleepapnea",
        "chronicbronchitis",
        "allergy",
        "kidney",
        "thyroid",
        "cancer",
        "liver",
        "urinary",
        "neuropathy",
        # mental health
        "depression",
        "anxiety",
        # hearing + vision
        "hearing",
        "vision",
    ]
    idx = (subjdf[medhist_cols[0]] == "[no]").values
    for cond in medhist_cols:
        idx = idx & (subjdf[cond] == "[no]").values
    return idx


def convert_to_list(s):
    print(s)
    try:
        s = s.strip()  # Remove leading/trailing whitespace
        s = s[1:-1].strip()  # Remove the surrounding brackets
        s = re.sub(r"\s+", ",", s)  # Replace any whitespace sequence with a comma
        return ast.literal_eval(f"[{s}]")  # Evaluate as a list
    except Exception as e:
        print(f"Error converting: {s}")
        return None  # or handle it in another way


if __name__ == "__main__":
    main()
