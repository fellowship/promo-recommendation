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
    "even_cohort": True,
    "response_sig_a": 10,
    "cross_response": False
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
PARAM_FH = ["dependent", "independent"]
PARAM_MFH = [0.01, 1, 10]
PARAM_NTRAIN = 10
OUT_RESULT_PATH = "./intermediate/cohort_var_xgb"
FIG_PATH = "./figs/cohort_var_xgb"
os.makedirs(OUT_RESULT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

#%% training
result_ls = []
for cvar, fh, mfh, itrain in tqdm(
    list(itt.product(PARAM_VAR, PARAM_FH, PARAM_MFH, range(PARAM_NTRAIN)))
):
    if fh == "dependent":
        feat_cols = ["user_f0", "user_f1", "camp_f0", "camp_f1"]
        fh_cohort = True
    elif fh == "independent":
        feat_cols = ["user_f0", "user_f1", "camp_f0", "camp_f1"]
        fh_cohort = False

    data, user_df, camp_df = generate_data(
        cohort_variances=cvar, fh_cohort=fh_cohort, magnify_hf=mfh, **PARAM_DATA
    )

    model = XGBClassifier(n_estimators=PARAM_NROUND, **PARAM_XGB)
    score = cross_validate(model, data[feat_cols], data["response"])["test_score"]
    score = pd.DataFrame(
        {
            "cohort_variance": cvar,
            "fh": fh,
            "mfh": mfh,
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
fig1 = px.violin(
    result,
    x="cohort_variance",
    y="score",
    color="fh",
    category_orders={"fh": ["dependent", "independent"]},
    points=False,
)
fig1.update_layout(legend_title="hidden feature dependence")
fig1.write_html(os.path.join(FIG_PATH, "scores_even_fh.html"))

fig2 = px.violin(
    result,
    x="cohort_variance",
    y="score",
    color="mfh",
    category_orders={"mfh": [0.01, 1, 10]},
    points=False,
)
fig2.update_layout(legend_title="hidden feature magnitude")
fig2.write_html(os.path.join(FIG_PATH, "scores_even_mfh.html"))
