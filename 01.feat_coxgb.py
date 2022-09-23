#%% imports and definitions
import itertools as itt
import os
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from routine.data_generation import generate_data
from routine.models import CohortXGB
import plotly.express as px

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
OUT_RESULT_PATH = "./intermediate/feat_coxgb"
FIG_PATH = "./figs/feat_coxgb"
os.makedirs(OUT_RESULT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

#%% training
result_ls = []
for cvar, pkey, itrain in tqdm(
    list(itt.product(PARAM_VAR, PARAM_MAP.keys(), range(PARAM_NTRAIN)))
):
    cohort_var = np.array([cvar, cvar, 0.1])
    data_train, user_df, camp_df = generate_data(
        cohort_variances=cohort_var, **PARAM_DATA
    )
    data_user, _, _ = generate_data(
        cohort_variances=cohort_var, user_df=user_df, **PARAM_DATA
    )
    data_test, _, _ = generate_data(cohort_variances=cohort_var, **PARAM_DATA)
    cur_param = PARAM_MAP[pkey]
    model = CohortXGB(n_cohort=PARAM_DATA["num_cohort"], **cur_param, **PARAM_XGB)
    model.fit(data_train)
    score = pd.DataFrame(
        [
            {
                "cohort_variance": cvar,
                "feats": pkey,
                "itrain": itrain,
                "score_test": model.score(data_test),
                "score_user": model.score(data_user),
            }
        ]
    )
    result_ls.append(score)
result = pd.concat(result_ls, ignore_index=True)
result.to_csv(os.path.join(OUT_RESULT_PATH, "result.csv"), index=False)

#%% plot result
data_map = {
    "score_train": "Training data",
    "score_valid": "Seen users, Seen campaigns",
    "score_user": "Seen users, New campaigns",
    "score_test": "New users, New campaigns",
}
result = (
    pd.read_csv(os.path.join(OUT_RESULT_PATH, "result.csv"))
    .melt(
        id_vars=["cohort_variance", "feats", "itrain"],
        var_name="data_type",
        value_name="score",
    )
    .replace({"data_type": data_map})
)
fig = px.box(
    result,
    x="cohort_variance",
    y="score",
    color="feats",
    facet_col="data_type",
    labels={"score": "CV Score", "cohort_variance": "Cohort Variance"},
)
fig.write_html(os.path.join(FIG_PATH, "New Users-cohort_mi.html"))
