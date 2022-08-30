#%% imports and definitions
import itertools as itt
import os
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import plotly.express as px
from tqdm.auto import tqdm
from xgboost import XGBClassifier

from routine.clustering import Kmeans_cluster
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
# PARAM_FEAT = [
#     "real cohort id + visible features",
#     "clustered cohort id + visible features",
#     "visible features",
#     "all features",
# ]
PARAM_FEAT = ["all features"]
PARAM_SAMP = ["random", "by_camp"]
PARAM_NTRAIN = 10
PARAM_FONT_SZ = {"font_size": 16, "title_font_size": 24, "legend_title_font_size": 24}
OUT_RESULT_PATH = "./intermediate/feat_samp_online"
FIG_PATH = "./figs/feat_samp_online"
os.makedirs(OUT_RESULT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

#%% training
result_ls = []
for cvar, feats, samp, itrain in tqdm(
    list(itt.product(PARAM_VAR, PARAM_FEAT, PARAM_SAMP, range(PARAM_NTRAIN)))
):
    data_train, _, _ = generate_data(cohort_variances=cvar, **PARAM_DATA)
    data_test, _, _ = generate_data(cohort_variances=cvar, **PARAM_DATA)
    if feats == "real cohort id + visible features":
        feat_cols = ["cohort", "user_f0", "user_f1", "camp_f0", "camp_f1"]
    elif feats == "clustered cohort id + visible features":
        feat_cols = ["cohort", "user_f0", "user_f1", "camp_f0", "camp_f1"]
        data_test["cohort"] = Kmeans_cluster(
            data_test[["user_f0", "user_f1"]], PARAM_DATA["num_cohort"]
        )
    elif feats == "visible features":
        feat_cols = ["user_f0", "user_f1", "camp_f0", "camp_f1"]
    elif feats == "all features":
        feat_cols = ["user_f0", "user_f1", "user_fh", "camp_f0", "camp_f1"]
    data_test_feat = data_test[feat_cols]
    if "cohort" in feat_cols:
        data_test_feat = pd.get_dummies(data_test_feat, columns=["cohort"])
    nsplit = PARAM_DATA["num_campaigns"]
    if samp == "random":
        idxs = np.array(data_train.index)
        np.random.shuffle(idxs)
        data_train_ls = [data_train.loc[i] for i in np.array_split(idxs, nsplit)]
    elif samp == "by_camp":
        data_train_ls = [g[1] for g in data_train.groupby("camp_id")]
    model = XGBClassifier(n_estimators=PARAM_NROUND, **PARAM_XGB)
    score = np.zeros(len(data_train_ls))
    for isub, cur_data in enumerate(data_train_ls):
        if feats == "clustered cohort id + visible features":
            cur_data["cohort"] = Kmeans_cluster(
                cur_data[["user_f0", "user_f1"]], PARAM_DATA["num_cohort"]
            )
        cur_feat = cur_data[feat_cols]
        if "cohort" in feat_cols:
            cur_feat = pd.get_dummies(cur_feat, columns=["cohort"])
        if isub > 0:
            model.fit(cur_feat, cur_data["response"], xgb_model=model.get_booster())
        else:
            model.fit(cur_feat, cur_data["response"])
        # score[isub] = model.score(data_test_feat, data_test["response"])
        score[isub] = model.score(data_train[feat_cols], data_train["response"])
    score = pd.DataFrame(
        {
            "cohort_variance": cvar,
            "feats": feats,
            "itrain": itrain,
            "data_prop": (np.arange(nsplit) + 1) / nsplit,
            "score": score,
            "sampling": samp,
        }
    )
    result_ls.append(score)
    break
result = pd.concat(result_ls, ignore_index=True)
# result.to_csv(os.path.join(OUT_RESULT_PATH, "result.csv"), index=False)

#%% plot result
result = pd.read_csv(os.path.join(OUT_RESULT_PATH, "result.csv"))
fig = px.line(result, x="data_prop", y="score")
fig.show()
#%%
fig = px.box(
    result,
    x="cohort_variance",
    y="score",
    color="cs",
    category_orders={
        "cs": ["cohort id", "numerical features", "cohort id + numerical features"]
    },
)
fig.update_layout(
    legend_title="Input to the model",
    xaxis_title="Cohort Variance",
    yaxis_title="CV Score",
    **PARAM_FONT_SZ
)
fig.write_html(os.path.join(FIG_PATH, "scores.html"))
