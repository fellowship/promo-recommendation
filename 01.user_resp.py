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
from routine.models import AddPCA
from routine.plotting import line
import plotly.express as px

PARAM_DATA = {
    "num_users": 1000,
    "num_campaigns": 30,
    "samples_per_campaign": 1000,
    "num_cohort": 10,
    "fh_cohort": True,
    "even_cohort": True,
    "response_sig_a": 10,
    "cross_weight": None,
    "magnify_hf": 1,
    "learning_rate_story": True
}
PARAM_XGB = {
    "max_depth": 5,
    "learning_rate": 1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "use_label_encoder": False,
}
PARAM_VAR = np.linspace(0.1, 0.6, 3)
PARAMS_PCA = [True, False]
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
    "response PCA + visible features": {
        "feats": ["user_f0", "user_f1", "camp_f0", "camp_f1", "camp_fh", "PCA_1", "PCA_2", "PCA_3"]
    },
    "visible features": {
        "feats": ["user_f0", "user_f1", "camp_f0", "camp_f1", "camp_fh"]
    },
    "all features": {
        "feats": ["user_f0", "user_f1", "user_fh", "camp_f0", "camp_f1", "camp_fh"]
    }
}
PARAM_NTRAIN = 20
PARAM_FONT_SZ = {"font_size": 16, "title_font_size": 24, "legend_title_font_size": 24}
OUT_RESULT_PATH = "./intermediate/user_resp"
FIG_PATH = "figs/user_resp"
os.makedirs(OUT_RESULT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

#%% training
result_ls = []
for cvar, pkey, itrain in tqdm(
    list(itt.product(PARAM_VAR, PARAM_MAP.keys(), range(PARAM_NTRAIN)))
):
    # Training data
    data_train, user_df, camp_df = generate_data(
        cohort_variances=cvar, **PARAM_DATA
    )
    # Validating against seen users and seen campaigns
    data_valid, _, _ = generate_data(
        cohort_variances=cvar, user_df=user_df, camp_df=camp_df, **PARAM_DATA
    )
    # Validating against seen users but new campaigns
    data_user, _, _ = generate_data(
        cohort_variances=cvar, user_df=user_df, **PARAM_DATA
    )

    cur_param = PARAM_MAP[pkey]

    if pkey == "response PCA + visible features":
        data_train, pca_df = AddPCA(data_train, n_campaigns=PARAM_DATA["num_campaigns"])
        data_valid = data_valid.merge(pca_df, how="left", on="user_id")
        data_user = data_user.merge(pca_df, how="left", on="user_id")

    nsplit = PARAM_DATA["num_campaigns"]
    data_train_ls = [g[1] for g in data_train.groupby("camp_id")]
    score_valid, score_user, score_train = (
        np.zeros(len(data_train_ls)),
        np.zeros(len(data_train_ls)),
        np.zeros(len(data_train_ls))
    )

    model = CohortXGB(n_cohort=PARAM_DATA["num_cohort"], **cur_param, **PARAM_XGB)
    for isub in range(len(data_train_ls)):
        cur_data = pd.concat(data_train_ls[:isub + 1], ignore_index=True)
        model.fit(cur_data)
        score_train[isub] = model.score(data_train)
        score_valid[isub] = model.score(data_valid)
        score_user[isub] = model.score(data_user)

    score = pd.DataFrame(
        {
            "cohort_variance": cvar,
            "feats": pkey,
            "itrain": itrain,
            "data_prop": (np.arange(nsplit) + 1) / nsplit,
            "score_valid": score_valid,
            "score_user": score_user,
            "score_train": score_train,
        }
    )
    result_ls.append(score)
result = pd.concat(result_ls, ignore_index=True)
result.to_csv(os.path.join(OUT_RESULT_PATH, "result.csv"), index=False)

#%% plot result by features
result = pd.read_csv(os.path.join(OUT_RESULT_PATH, "result.csv"))
os.makedirs(os.path.join(FIG_PATH, "by_feats"), exist_ok=True)
title_map = {
    "score_valid": "Seen users, Seen campaigns",
    "score_user": "Seen users, New campaigns",
    "score_train": "Training data",
}
for sc_name, plt_title in title_map.items():
    data_df = (
        result.groupby(["cohort_variance", "feats", "data_prop"])[sc_name]
        .agg(["mean", "sem"])
        .reset_index()
    )
    fig = line(
        data_frame=data_df,
        x="data_prop",
        y="mean",
        color="feats",
        facet_col="cohort_variance",
        error_y="sem",
        error_y_mode="bands",
        labels={"data_prop": "Proportion of<br>Data/Campaigns", "mean": "CV Score"},
        range_y=(0.35, 0.9),
    )
    fig.update_layout(
        title={"text": plt_title, "x": 0.5, "xanchor": "center"},
        legend_title="features",
        **PARAM_FONT_SZ,
    )
    fig.write_html(os.path.join(FIG_PATH, "by_feats", "{}.html".format(sc_name)))

#%% plot result by score
result = pd.read_csv(os.path.join(OUT_RESULT_PATH, "result.csv"))
data_map = {
    "score_train": "Training data",
    "score_valid": "Seen users, Seen campaigns",
    "score_user": "Seen users, New campaigns"
}
id_cols = result.columns[
    list(map(lambda c: not c.startswith("score_"), result.columns))
]
result = (
    result.melt(id_vars=id_cols, var_name="data_type", value_name="score")
    .replace({"data_type": data_map})
    .groupby(["cohort_variance", "feats", "data_prop", "data_type"])[
        "score"
    ]
    .agg(["mean", "sem"])
    .reset_index()
)
os.makedirs(os.path.join(FIG_PATH, "by_score"), exist_ok=True)
for feat, data_df in result.groupby("feats"):
    fig = line(
        data_frame=data_df,
        x="data_prop",
        y="mean",
        color="data_type",
        facet_col="cohort_variance",
        error_y="sem",
        error_y_mode="bands",
        labels={"data_prop": "Proportion of<br>Data/Campaigns", "mean": "CV Score"},
        range_y=(0.45, 1),
        category_orders={"data_type": list(data_map.values())},
    )
    fig.update_layout(
        title={"text": feat, "x": 0.5, "xanchor": "center"},
        legend_title="Testing Data",
        **PARAM_FONT_SZ,
    )
    fig.write_html(os.path.join(FIG_PATH, "by_score", "{}.html".format(feat)))

#%% Plot the score of the asymptote
os.makedirs(os.path.join(FIG_PATH, "by_score"), exist_ok=True)
result = pd.read_csv(os.path.join(OUT_RESULT_PATH, "result.csv"))
tmp = [g[1] for g in result.groupby("data_prop")]
result = tmp[-1]

title_map = {
    "score_valid": "Seen users, Seen campaigns",
    "score_user": "Seen users, New campaigns",
    "score_train": "Training data",
}
for sc_name, plt_title in title_map.items():
    fig = px.box(
        result,
        x="cohort_variance",
        y=sc_name,
        color="feats",
        category_orders={
            "feats": ["real cohort id + visible features",
                      "clustered cohort id + visible features",
                      "response-clustered cohort id + visible features",
                      "response PCA + visible features",
                      "visible features",
                      "all features"]
        },
    )
    fig.update_layout(
        title={"text": plt_title, "x": 0.5, "xanchor": "center"},
        legend_title="Input to the model",
        xaxis_title="Cohort Variance",
        yaxis_title="CV Score",
        **PARAM_FONT_SZ
    )
    fig.write_html(os.path.join(FIG_PATH, "by_score", "{}.html".format(sc_name)))

