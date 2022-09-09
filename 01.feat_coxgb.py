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
from routine.plotting import scatter_3d
from plotly.express.colors import qualitative

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
    "real cohort id": {
        "feats": ["cohort", "user_f0", "user_f1", "camp_f0", "camp_f1", "camp_fh"]
    },
    "visible-clustered cohort id": {
        "feats": ["cohort", "user_f0", "user_f1", "camp_f0", "camp_f1", "camp_fh"],
        "cohort_feats": ["user_f0", "user_f1"],
        "use_cohort_resp": False,
    },
    "response-clustered cohort id": {
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

#%% case study of cohort accuracy
def plot_users(df, cmap, z="user_fh", col="cohort"):
    df[col + "_col"] = df[col].map(cmap)
    fig = scatter_3d(
        df.astype({col: str}),
        x="user_f0",
        y="user_f1",
        z=z,
        mode="markers",
        marker={"color": col + "_col"},
        legend_dim=col,
        facet_col="visible_variance",
        facet_row=None,
    )
    fig.update_traces(marker_size=3)
    return fig


vis_var = {"low": 0.1, "high": 0.6}
var_fh = 0.1
means = np.array([[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]])
param_data = {
    "num_users": 1000,
    "num_campaigns": 1,
    "samples_per_campaign": 2000,
    "num_cohort": 2,
    "fh_cohort": True,
    "even_cohort": True,
    "response_sig_a": 10,
    "cross_weight": None,
    "magnify_hf": 1,
}
param_model = {
    "n_cohort": 2,
    "feats": ["cohort", "user_f0", "user_f1", "camp_f0", "camp_f1", "camp_fh"],
    "cohort_feats": ["user_f0", "user_f1"],
    "max_depth": 5,
    "learning_rate": 1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "use_label_encoder": False,
}
camp_df = pd.DataFrame(
    [
        {
            "camp_id": 0,
            "freq": param_data["samples_per_campaign"],
            "camp_f0": 1,
            "camp_f1": 1,
            "camp_fh": 1,
        }
    ]
)
ax_title_feat = {
    "xaxis_title": "Feature 1",
    "yaxis_title": "Feature 2",
    "zaxis_title": "Hidden Feature",
}
ax_title_resp = {
    "xaxis_title": "Feature 1",
    "yaxis_title": "Feature 2",
    "zaxis_title": "Response",
}
cmap_real = {0: qualitative.Plotly[0], 1: qualitative.Plotly[1]}
cmap_prd = {0: qualitative.Plotly[2], 1: qualitative.Plotly[3]}
data_df = []
user_df = []
for var_lev, cur_var in vis_var.items():
    cur_data, cur_user, _ = generate_data(
        cohort_means=means,
        cohort_variances=np.array([cur_var, cur_var, var_fh]),
        camp_df=camp_df,
        **param_data,
    )
    model = CohortXGB(use_cohort_resp=False, **param_model)
    model.fit(cur_data)
    cur_data["cohort_cluster"] = model.predict_cohort(cur_data)
    model = CohortXGB(use_cohort_resp=True, **param_model)
    model.fit(cur_data)
    cur_data["cohort_resp"] = model.predict_cohort(cur_data)
    cur_data["visible_variance"] = var_lev
    cur_user["visible_variance"] = var_lev
    data_df.append(cur_data)
    user_df.append(cur_user)
data_df = pd.concat(data_df, ignore_index=True)
user_df = pd.concat(user_df, ignore_index=True)
# plot user features colored by real cohort
fig_user = plot_users(user_df, cmap_real)
fig_user.update_layout(
    legend={"title": "Cohort", "itemsizing": "constant"},
    title="Real Cohorts",
    scene=ax_title_feat,
    scene2=ax_title_feat,
    **PARAM_FONT_SZ,
)
fig_user.write_html(os.path.join(FIG_PATH, "user_real_cohort.html"))
# plot clustering without response
fig_cluster = plot_users(data_df, cmap_prd, col="cohort_cluster")
fig_cluster.update_layout(
    legend={"title": "Predicted Cohort", "itemsizing": "constant"},
    title="Visible Clustered Cohorts",
    scene=ax_title_feat,
    scene2=ax_title_feat,
    **PARAM_FONT_SZ,
)
fig_cluster.write_html(os.path.join(FIG_PATH, "cluster_cohort.html"))
# plot clustering with response
fig_resp = plot_users(data_df, cmap_real, z="response", col="cohort")
fig_resp.update_layout(
    legend={"title": "Cohort", "itemsizing": "constant"},
    title="Augmented Space",
    scene=ax_title_resp,
    scene2=ax_title_resp,
    **PARAM_FONT_SZ,
)
fig_resp.write_html(os.path.join(FIG_PATH, "resp_cohort.html"))
# plot clustering with response
fig_prd = plot_users(data_df, cmap_prd, col="cohort_resp")
fig_prd.update_layout(
    legend={"title": "Predicted Cohort", "itemsizing": "constant"},
    title="Response Clustered Cohorts",
    scene=ax_title_resp,
    scene2=ax_title_resp,
    **PARAM_FONT_SZ,
)
fig_prd.write_html(os.path.join(FIG_PATH, "prd_cohort.html"))
