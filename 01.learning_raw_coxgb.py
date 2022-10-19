#%% imports and definitions
import itertools as itt
import os
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.express.colors import qualitative
from sklearn.metrics import normalized_mutual_info_score
from tqdm.auto import tqdm

from routine.data_generation import generate_data
from routine.models import CohortXGB
from routine.plotting import line
from routine.training import cv_by_id

PARAM_DATA = {
    "num_users": 1000,
    "num_campaigns": 100,
    "samples_per_campaign": 1000,
    "num_cohort": 40,
    "fh_cohort": True,
    "even_cohort": True,
    "response_sig_a": 10,
    "cross_weight": None,
    "magnify_hf": 2,
    "perfect_camp": True,
}
PARAM_XGB = {
    "max_depth": 5,
    "learning_rate": 1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "use_label_encoder": False,
}
PARAM_SUB_CAMP = 5
PARAM_NROUND = 30
PARAM_VAR = np.linspace(0.1, 0.9, 3)
PARAM_MAP = {
    "raw response": {
        "feats": ["user_f0", "user_f1", "camp_f0", "camp_f1", "camp_fh"],
        "user_feats": ["user_f0", "user_f1"],
        "use_raw_resp": True,
    },
    "real cohort id": {
        "feats": ["cohort", "user_f0", "user_f1", "camp_f0", "camp_f1", "camp_fh"]
    },
    "response-clustered cohort id": {
        "feats": ["cohort", "user_f0", "user_f1", "camp_f0", "camp_f1", "camp_fh"],
        "cohort_feats": ["user_f0", "user_f1"],
        "user_feats": ["user_f0", "user_f1"],
        "use_cohort_resp": True,
    },
    "visible features": {
        "feats": ["user_f0", "user_f1", "camp_f0", "camp_f1", "camp_fh"]
    },
    "all features": {
        "feats": ["user_f0", "user_f1", "user_fh", "camp_f0", "camp_f1", "camp_fh"]
    },
}
PARAM_NTRAIN = 10
PARAM_FONT_SZ = {"font_size": 16, "title_font_size": 24, "legend_title_font_size": 24}
OUT_RESULT_PATH = "./intermediate/learning_raw_coxgb"
FIG_PATH = "./figs/learning_raw_coxgb"
os.makedirs(OUT_RESULT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

#%% training
result_ls = []
for cvar, pkey, itrain in tqdm(
    list(itt.product(PARAM_VAR, PARAM_MAP.keys(), range(PARAM_NTRAIN)))
):
    if pkey != "response-clustered cohort id":
        continue
    cohort_var = np.array([cvar, cvar, 0.1])
    data, user_df, camp_df = generate_data(cohort_variances=cohort_var, **PARAM_DATA)
    user_df_new = user_df.copy()
    user_df_new["user_fh"] = user_df_new["user_fh"] * -1
    data_new_train, _, _ = generate_data(
        cohort_variances=cohort_var, user_df=user_df_new, **PARAM_DATA
    )
    data_new_test, _, _ = generate_data(
        cohort_variances=cohort_var, user_df=user_df_new, **PARAM_DATA
    )
    cur_param = PARAM_MAP[pkey]
    model = CohortXGB(n_cohort=PARAM_DATA["num_cohort"], **cur_param, **PARAM_XGB)
    model.fit(data)
    data_train_ls = [g[1] for g in data_new_train.groupby("camp_id")][:PARAM_SUB_CAMP]
    scores_learn = np.full(len(data_train_ls), np.nan)
    for isub in range(len(data_train_ls)):
        cur_data = pd.concat(data_train_ls[: isub + 1], ignore_index=True)
        for _ in range(10):
            model.fit_xgb(cur_data, warm_start=True)
        scores_learn[isub] = model.score(data_new_test)
    score = pd.DataFrame(
        {
            "cohort_variance": cvar,
            "feats": pkey,
            "itrain": itrain,
            "ncamp": np.arange(len(data_train_ls)),
            "scores_learn": scores_learn,
        }
    )
    result_ls.append(score)
    break
# result = pd.concat(result_ls, ignore_index=True)
# result.to_csv(os.path.join(OUT_RESULT_PATH, "result.csv"), index=False)

#%% plot result
result = pd.read_csv(os.path.join(OUT_RESULT_PATH, "result.csv"))
result_agg = (
    result.groupby(["cohort_variance", "feats", "ncamp"])["scores_learn"]
    .agg(["mean", "sem"])
    .reset_index()
)
fig = line(
    data_frame=result_agg,
    x="ncamp",
    y="mean",
    facet_row="cohort_variance",
    color="feats",
    error_y="sem",
    error_y_mode="bands",
    labels={"mean": "Test Score", "ncamp": "Number of campaigns"},
)
fig.update_layout(legend={"title": None}, **PARAM_FONT_SZ)
fig.write_html(os.path.join(FIG_PATH, "scores.html"))
fig.show()
