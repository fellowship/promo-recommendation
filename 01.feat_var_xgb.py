#%% imports and definitions
import itertools as itt
import os
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.express.colors import qualitative
from plotly.subplots import make_subplots
from sklearn.model_selection import cross_validate
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
PARAM_FONT_SZ = {"font_size": 16, "title_font_size": 24, "legend_title_font_size": 24}
PARAM_NROUND = 30
PARAM_VAR_F = np.linspace(0.1, 0.6, 6)
PARAM_VAR_FH = np.linspace(0.1, 0.6, 6)
PARAM_COHORT = [
    "real cohort id + numerical features",
    "clustered cohort id + numerical features",
    "numerical features",
    "all features",
]
PARAM_NTRAIN = 3
OUT_RESULT_PATH = "./intermediate/feat_var_xgb"
FIG_PATH = "./figs/feat_var_xgb"
os.makedirs(OUT_RESULT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

#%% training
result_ls = []
for var_f, var_fh, cs, itrain in tqdm(
    list(itt.product(PARAM_VAR_F, PARAM_VAR_FH, PARAM_COHORT, range(PARAM_NTRAIN)))
):
    data, user_df, camp_df = generate_data(
        cohort_variances=np.array([var_f, var_f, var_fh]), **PARAM_DATA
    )
    if cs == "real cohort id + numerical features":
        feat_cols = ["cohort", "user_f0", "user_f1", "camp_f0", "camp_f1"]
    elif cs == "clustered cohort id + numerical features":
        feat_cols = ["cohort", "user_f0", "user_f1", "camp_f0", "camp_f1"]
        data["cohort"] = Kmeans_cluster(
            data[["user_f0", "user_f1"]], PARAM_DATA["num_cohort"]
        )
    elif cs == "numerical features":
        feat_cols = ["user_f0", "user_f1", "camp_f0", "camp_f1"]
    elif cs == "all features":
        feat_cols = ["user_f0", "user_f1", "user_fh", "camp_f0", "camp_f1"]
    data_feat = data[feat_cols]
    if "cohort" in feat_cols:
        data_feat = pd.get_dummies(data_feat, columns=["cohort"])
    model = XGBClassifier(n_estimators=PARAM_NROUND, **PARAM_XGB)
    score = cross_validate(model, data_feat, data["response"])["test_score"]
    score = pd.DataFrame(
        {
            "var_fvis": var_f,
            "var_fh": var_fh,
            "cs": cs,
            "itrain": itrain,
            "cv": np.arange(len(score)),
            "score": score,
        }
    )
    result_ls.append(score)
result = pd.concat(result_ls, ignore_index=True)
result.to_csv(os.path.join(OUT_RESULT_PATH, "result.csv"), index=False)

#%% plot 3d surface
result = pd.read_csv(os.path.join(OUT_RESULT_PATH, "result.csv"))
result_agg = result.groupby(["var_fvis", "var_fh", "cs"])["score"].mean().reset_index()
options_all = {"opacity": 1, "showscale": False, "showlegend": True}
options = {
    "numerical features": {
        "colorscale": "blues_r",
        "name": "numerical features",
        "legendgroup": "numerical features",
    },
    "all features": {
        "colorscale": "greens_r",
        "opacity": 0.9,
        "name": "all features",
        "legendgroup": "all features",
    },
    "real cohort id + numerical features": {
        "colorscale": "reds_r",
        "name": "real cohort id + numerical features",
        "legendgroup": "cohort id + numerical features",
    },
    "clustered cohort id + numerical features": {
        "colorscale": "purples_r",
        "name": "clustered cohort id + numerical features",
        "legendgroup": "cohort id + numerical features",
    },
}
scene_opts = {
    "aspectmode": "cube",
    "xaxis_title": "hidden features variance",
    "yaxis_title": "visible feature variance",
    "zaxis_title": "CV score",
}
fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"is_3d": True}, {"is_3d": True}]],
    subplot_titles=[
        "<b>real cohort id</b> + numerical features",
        "<b>clustered cohort id</b> + numerical features",
    ],
)
for cs, cs_df in result_agg.groupby("cs"):
    arr = cs_df.set_index(["var_fvis", "var_fh"])["score"].to_xarray()
    opts = options_all | options.get(cs, {})
    trace = go.Surface(
        x=arr.coords["var_fvis"],
        y=arr.coords["var_fh"],
        z=arr.values,
        **opts,
    )
    if cs == "real cohort id + numerical features":
        fig.add_trace(trace, row=1, col=1)
    elif cs == "clustered cohort id + numerical features":
        fig.add_trace(trace, row=1, col=2)
    else:
        for c in range(2):
            fig.add_trace(trace, row=1, col=c + 1)
fig.update_layout(scene=scene_opts, scene2=scene_opts, **PARAM_FONT_SZ)
fig.update_annotations(font_size=PARAM_FONT_SZ["title_font_size"])
fig.write_html(os.path.join(FIG_PATH, "3d_score.html"))

#%% plot slice of score
result = pd.read_csv(os.path.join(OUT_RESULT_PATH, "result.csv"))
result_sub = result[result["var_fh"] == 0.2]
fig = px.box(
    result_sub,
    x="var_fvis",
    y="score",
    color="cs",
    category_orders={
        "cs": [
            "all features",
            "real cohort id + numerical features",
            "clustered cohort id + numerical features",
            "numerical features",
        ]
    },
    color_discrete_map={
        "all features": qualitative.Plotly[2],
        "real cohort id + numerical features": qualitative.Plotly[1],
        "clustered cohort id + numerical features": qualitative.Plotly[3],
        "numerical features": qualitative.Plotly[0],
    },
)
fig.update_layout(
    xaxis_title="Visible Feature Variance",
    yaxis_title="CV Score",
    legend_title="Input to the model",
    title="hidden feature variance: 0.2",
    **PARAM_FONT_SZ,
)
fig.write_html(os.path.join(FIG_PATH, "score_slice.html"))
