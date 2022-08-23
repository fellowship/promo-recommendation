#%% imports and definitions
import itertools as itt
import os
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import cross_validate
from tqdm.auto import tqdm
from xgboost import XGBClassifier

from routine.data_generation import generate_data

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
    "var_fac_hf": 1
}
PARAM_XGB = {
    "max_depth": 5,
    "learning_rate": 1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "use_label_encoder": False,
}
PARAM_NROUND = 30
PARAM_VAR = np.linspace(0.05, 0.6, 12)
PARAM_COHORT = ["cohort id", "numerical features", "cohort id + numerical features"]
PARAM_NTRAIN = 10
OUT_RESULT_PATH = "./intermediate/cohort_feat_xgb"
FIG_PATH = "./figs/cohort_feat_xgb"
os.makedirs(OUT_RESULT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

#%% training
result_ls = []
for cvar, cs, itrain in tqdm(
    list(itt.product(PARAM_VAR, PARAM_COHORT, range(PARAM_NTRAIN)))
):
    data, user_df, camp_df = generate_data(
        cohort_variances=cvar, **PARAM_DATA
    )

    if cs == "cohort id":
        feat_cols = ["cohort", "camp_f0", "camp_f1"]
        data_modified = pd.get_dummies(data[feat_cols], columns=['cohort'])
    elif cs == "numerical features":
        feat_cols = ["user_f0", "user_f1", "camp_f0", "camp_f1"]
        data_modified = data[feat_cols]
    elif cs == "cohort id + numerical features":
        feat_cols = ["cohort", "user_f0", "user_f1", "camp_f0", "camp_f1"]
        data_modified = pd.get_dummies(data[feat_cols], columns=["cohort"])

    model = XGBClassifier(n_estimators=PARAM_NROUND, **PARAM_XGB)
    score = cross_validate(model, data_modified, data["response"])["test_score"]
    score = pd.DataFrame(
        {
            "cohort_variance": cvar,
            "cs": cs,
            "itrain": itrain,
            "cv": np.arange(len(score)),
            "score": score,
        }
    )
    result_ls.append(score)
result = pd.concat(result_ls, ignore_index=True)
result.to_csv(os.path.join(OUT_RESULT_PATH, "result.csv"), index=False)

#%% plotting
result = pd.read_csv(os.path.join(OUT_RESULT_PATH, "result.csv"))
fig = px.box(
    result,
    x="cohort_variance",
    y="score",
    color="cs",
    category_orders={"cs": ["cohort id", "numerical features", "cohort id + numerical features"]}
)
fig.update_layout(legend_title="Input to the model")

fig.update_layout(
    font=dict(
        family="Courier New, monospace",
        size=15,
        color="RebeccaPurple"
    ),
    category_orders={
        "cs": ["only_cohort", "only_real_features", "cohort_and_real_features"]
    }
)
fig.write_html(os.path.join(FIG_PATH, "scores_even_cohort_importance.html"))
