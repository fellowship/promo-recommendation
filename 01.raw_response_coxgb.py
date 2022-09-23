#%% imports and definitions
import itertools as itt
import os
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm

from routine.data_generation import generate_data
from routine.models import CohortXGB
from routine.training import cv_by_id

PARAM_DATA = {
    "num_users": 1000,
    "num_campaigns": 100,
    "samples_per_campaign": 2000,
    "num_cohort": 10,
    "fh_cohort": True,
    "even_cohort": True,
    "response_sig_a": 10,
    "cross_weight": None,
    "magnify_hf": 1,
}
PARAM_XGB = {
    "max_depth": 6,
    "learning_rate": 1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "use_label_encoder": False,
}
PARAM_NROUND = 30
PARAM_VAR = [0.1, 1.0]
PARAM_MAP = {
    "raw response - double campaign": {
        "feats": ["user_f0", "user_f1", "camp_f0", "camp_f1", "camp_fh"],
        "user_feats": ["user_f0", "user_f1"],
        "use_raw_resp": True,
    },
    "raw response": {
        "feats": ["user_f0", "user_f1", "camp_f0", "camp_f1", "camp_fh"],
        "user_feats": ["user_f0", "user_f1"],
        "use_raw_resp": True,
    },
    "visible features": {
        "feats": ["user_f0", "user_f1", "camp_f0", "camp_f1", "camp_fh"]
    },
}
PARAM_NTRAIN = 20
PARAM_FONT_SZ = {"font_size": 16, "title_font_size": 24, "legend_title_font_size": 24}
PARAM_CV = 5
PARAM_SPLT_BY = ["camp_id"]
OUT_RESULT_PATH = "./intermediate/raw_coxgb"
FIG_PATH = "./figs/raw_coxgb"
os.makedirs(OUT_RESULT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

#%% training
result_ls = []
for cvar, pkey, splt_by, itrain in tqdm(
    list(itt.product(PARAM_VAR, PARAM_MAP.keys(), PARAM_SPLT_BY, range(PARAM_NTRAIN)))
):
    cohort_var = np.array([cvar, cvar, 0.1])
    cur_param_data = PARAM_DATA.copy()
    if pkey == "raw response - double campaign":
        cur_param_data["num_campaigns"] = cur_param_data["num_campaigns"] * 2
    data, _, _ = generate_data(cohort_variances=cohort_var, **cur_param_data)
    cur_param = PARAM_MAP[pkey]
    scores_train, scores_test = np.full(PARAM_CV, np.nan), np.full(PARAM_CV, np.nan)
    for icv, (data_train, data_test) in enumerate(cv_by_id(data, PARAM_CV, splt_by)):
        model = CohortXGB(n_cohort=PARAM_DATA["num_cohort"], **cur_param, **PARAM_XGB)
        model.fit(data_train)
        scores_train[icv] = model.score(data_train)
        scores_test[icv] = model.score(data_test)
    score = pd.DataFrame(
        {
            "cohort_variance": cvar,
            "feats": pkey,
            "itrain": itrain,
            "split_by": splt_by,
            "cv": np.arange(PARAM_CV),
            "scores_train": scores_train,
            "scores_test": scores_test,
        }
    )
    result_ls.append(score)
result = pd.concat(result_ls, ignore_index=True)
result.to_csv(os.path.join(OUT_RESULT_PATH, "result.csv"), index=False)

#%% plot result
result = (
    pd.read_csv(os.path.join(OUT_RESULT_PATH, "result.csv"))
    .melt(
        id_vars=["cohort_variance", "feats", "itrain", "cv"],
        value_vars=["scores_train", "scores_test"],
        var_name="score_type",
        value_name="scores",
    )
    .replace({"score_type": {"scores_train": "training", "scores_test": "testing"}})
)
ord_ls = ["raw response", "raw response - double campaign", "visible features"]
fig = px.box(
    result,
    x="cohort_variance",
    y="scores",
    color="feats",
    labels={
        "scores": "CV Score",
        "cohort_variance": "Visible Feature Variance",
        "cohort_mi": "Mutual information with<br>real cohort ID",
    },
    facet_col="score_type",
    category_orders={"feats": ord_ls},
)
fig.update_layout(legend={"title": None}, **PARAM_FONT_SZ)
fig.write_html(os.path.join(FIG_PATH, "scores.html"))
