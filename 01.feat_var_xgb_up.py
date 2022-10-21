#%% imports and definitions
import itertools as itt
import os
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from plotly.express.colors import qualitative
from routine.data_generation import generate_data
from routine.models import CohortXGB
from routine.plotting import line
import plotly.express as px

PARAM_DATA = {
    "num_users": 1000,
    "num_campaigns": 50,
    "samples_per_campaign": 1000,
    "num_cohort": 40,
    "fh_cohort": True,
    "even_cohort": True,
    "response_sig_a": 10,
    "cross_weight": None,
    "magnify_hf": 2,
    "learning_rate_story": True
}
PARAM_XGB = {
    "max_depth": 5,
    "learning_rate": 1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "use_label_encoder": False,
}
PARAM_NROUND = 30
PARAM_VAR = np.linspace(0.1, 1, 10)
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
OUT_RESULT_PATH = "./intermediate/feat_var_xgb_up"
FIG_PATH = "figs/feat_var_xgb_up"
os.makedirs(OUT_RESULT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

#%% training
result_ls = []
for cvar, pkey, itrain in tqdm(
    list(itt.product(PARAM_VAR, PARAM_MAP.keys(), range(PARAM_NTRAIN)))
):
    # Training data
    data_train, user_df, camp_df = generate_data(
        cohort_variances=np.array([cvar, cvar, 0.1]), **PARAM_DATA
    )
    # Validating against seen users and seen campaigns
    data_valid, _, _ = generate_data(
        cohort_variances=np.array([cvar, cvar, 0.1]), user_df=user_df, camp_df=camp_df, **PARAM_DATA
    )
    # Validating against seen users but new campaigns
    data_user, _, _ = generate_data(
        cohort_variances=np.array([cvar, cvar, 0.1]), user_df=user_df, **PARAM_DATA
    )
    # Validating against new users and new campaigns
    data_test, _, _ = generate_data(
        cohort_variances=np.array([cvar, cvar, 0.1]), **PARAM_DATA
    )

    cur_param = PARAM_MAP[pkey]

    model = CohortXGB(n_cohort=PARAM_DATA["num_cohort"], **cur_param, **PARAM_XGB)
    model.fit(data_train)
    score_train = model.score(data_train)
    score_valid = model.score(data_valid)
    score_user = model.score(data_user)
    score_test = model.score(data_test)

    score = pd.DataFrame(
        [
            {
                "cohort_variance": cvar,
                "feats": pkey,
                "itrain": itrain,
                "score_test": score_test,
                "score_valid": score_valid,
                "score_user": score_user,
                "score_train": score_train,
            }
        ]
    )
    result_ls.append(score)
result = pd.concat(result_ls, ignore_index=True)
result.to_csv(os.path.join(OUT_RESULT_PATH, "result_up.csv"), index=False)

#%% plot result by score
result = pd.read_csv(os.path.join(OUT_RESULT_PATH, "result_up.csv"))
fig = px.box(
    result,
    x="cohort_variance",
    y="score_user",
    color="feats",
    category_orders={
        "feats": [
            "all features",
            "real cohort id + visible features",
            "clustered cohort id + visible features",
            "response-clustered cohort id + visible features"
            "visible features",
        ]
    },
    color_discrete_map={
        "all features": qualitative.Plotly[2],
        "real cohort id + visible features": qualitative.Plotly[1],
        "clustered cohort id + visible features": qualitative.Plotly[3],
        "response-clustered cohort id + visible features": qualitative.Plotly[4],
        "visible features": qualitative.Plotly[0],
    },
)
fig.update_layout(
    xaxis_title="Visible Feature Variance",
    yaxis_title="CV Score",
    legend_title="Input to the model",
    **PARAM_FONT_SZ,
)
fig.write_html(os.path.join(FIG_PATH, "score_up.html"))
fig.write_image(os.path.join(FIG_PATH, "score_up.png"), width=1500, height=800, scale=1)
