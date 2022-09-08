#%% imports and definitions
import itertools as itt
import os
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import normalized_mutual_info_score
from tqdm.auto import tqdm

from routine.data_generation import generate_data
from routine.models import CohortXGB
from routine.training import cv_by_id

PARAM_DATA = {
    "num_users": 1000,
    "num_campaigns": 100,
    "samples_per_campaign": 10000,
    "num_cohort": 10,
    "fh_cohort": True,
    "even_cohort": True,
    "response_sig_a": 10,
    "cross_weight": None,
    "magnify_hf": 1,
}
PARAM_XGB = {
    "max_depth": 5,
    "learning_rate": 1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "use_label_encoder": False,
}
PARAM_NROUND = 30
PARAM_VAR = np.linspace(0.1, 0.6, 3)
PARAM_MAP = {
    "real cohort id + visible features": {
        "feats": ["cohort", "user_f0", "user_f1", "camp_f0", "camp_f1", "camp_fh"]
    },
    "clustered cohort id + visible features": {
        "feats": ["cohort", "user_f0", "user_f1", "camp_f0", "camp_f1", "camp_fh"],
        "cohort_feats": ["user_f0", "user_f1"],
        "use_cohort_resp": False,
    },
    "response-clustered cohort id + visible features": {
        "feats": ["cohort", "user_f0", "user_f1", "camp_f0", "camp_f1", "camp_fh"],
        "cohort_feats": ["user_f0", "user_f1"],
        "use_cohort_resp": True,
    },
    "visible features": {
        "feats": ["user_f0", "user_f1", "camp_f0", "camp_f1", "camp_fh"]
    },
    "all features": {
        "feats": ["user_f0", "user_f1", "user_fh", "camp_f0", "camp_f1", "camp_fh"]
    },
}
PARAM_NTRAIN = 20
PARAM_FONT_SZ = {"font_size": 16, "title_font_size": 24, "legend_title_font_size": 24}
PARAM_CV = 5
PARAM_SPLT_BY = ["camp_id", "user_id"]
OUT_RESULT_PATH = "./intermediate/feat_coxgb"
FIG_PATH = "./figs/feat_coxgb"
os.makedirs(OUT_RESULT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

#%% training
result_ls = []
for cvar, pkey, splt_by, itrain in tqdm(
    list(itt.product(PARAM_VAR, PARAM_MAP.keys(), PARAM_SPLT_BY, range(PARAM_NTRAIN)))
):
    cohort_var = np.array([cvar, cvar, 0.1])
    data, _, _ = generate_data(cohort_variances=cohort_var, **PARAM_DATA)
    cur_param = PARAM_MAP[pkey]
    scores, cohort_mi = np.full(PARAM_CV, np.nan), np.full(PARAM_CV, np.nan)
    for icv, (data_train, data_test) in enumerate(cv_by_id(data, PARAM_CV, splt_by)):
        model = CohortXGB(n_cohort=PARAM_DATA["num_cohort"], **cur_param, **PARAM_XGB)
        model.fit(data_train)
        if "cohort_feats" in cur_param:
            cohort_prd = model.predict_cohort(data_test)
            cohort_mi[icv] = normalized_mutual_info_score(
                data_test["cohort"], cohort_prd
            )
        scores[icv] = model.score(data_test)
    score = pd.DataFrame(
        {
            "cohort_variance": cvar,
            "feats": pkey,
            "itrain": itrain,
            "split_by": splt_by,
            "cv": np.arange(PARAM_CV),
            "scores": scores,
            "cohort_mi": cohort_mi,
        }
    )
    result_ls.append(score)
result = pd.concat(result_ls, ignore_index=True)
result.to_csv(os.path.join(OUT_RESULT_PATH, "result.csv"), index=False)

#%% plot result
result = pd.read_csv(os.path.join(OUT_RESULT_PATH, "result.csv"))
for yvar in ["scores", "cohort_mi"]:
    fig = px.box(
        result,
        x="cohort_variance",
        y=yvar,
        color="feats",
        facet_row="split_by",
        labels={"scores": "CV Score", "cohort_variance": "Visible Feature Variance"},
    )
    fig.write_html(os.path.join(FIG_PATH, "{}.html".format(yvar)))
