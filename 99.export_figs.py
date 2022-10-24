#%% imports and definition
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.express.colors import qualitative
from plotly.subplots import make_subplots

from routine.data_generation import generate_data
from routine.models import CohortXGB

IN_RESULT = "./intermediate/feat_coxgb/result.csv"
IN_MI = "./intermediate/mi_coxgb/mi_df.csv"
FIG_PATH = "./figs/export"
PARAM_DATA = {
    "num_users": 1000,
    "num_campaigns": 80,
    "samples_per_campaign": 1000,
    "num_cohort": 10,
    "fh_cohort": True,
    "even_cohort": True,
    "response_sig_a": 10,
    "cross_weight": None,
    "magnify_hf": 1,
    "perfect_camp": True,
}
PARAM_CMAP_FEAT = {
    "visible features": qualitative.Plotly[0],
    "real cohort id": qualitative.Plotly[1],
    "response-clustered cohort id": qualitative.Plotly[3],
    "raw response": qualitative.Plotly[4],
    "all features": qualitative.Plotly[2],
}
PARAM_NMAP_FEAT = {
    "visible features": "Visible Features",
    "real cohort id": "Real Cohort ID",
    "response-clustered cohort id": "Predicted Cohort ID",
    "raw response": "Raw Response",
    "all features": "All Features",
}
PARAM_CMAP_MI = {
    "prd_cohort": qualitative.Plotly[3],
    "real_cohort": qualitative.Plotly[1],
    "user_f0": qualitative.Plotly[0],
    "user_f1": qualitative.Plotly[5],
}
PARAM_NMAP_MI = {
    "prd_cohort": "Predicted Cohort ID",
    "real_cohort": "Real Cohort ID",
    "user_f0": "Visible Feature 1",
    "user_f1": "Visible Feature 2",
}
PARAM_VAR_SLC = 0.1
PARAM_FONT_SZ = {"font_size": 10, "title_font_size": 24, "legend_title_font_size": 24}


def feat_order(feat):
    ford = list(PARAM_CMAP_FEAT)
    return list(map(lambda f: ford.index(f), feat))


#%% fig 1
# defs
fig_path = os.path.join(FIG_PATH, "fig1")
os.makedirs(fig_path, exist_ok=True)
# generate data
_, user_df_low, camp_df = generate_data(cohort_variances=0.1, **PARAM_DATA)
_, user_df_high, camp_df = generate_data(cohort_variances=0.3, **PARAM_DATA)
# make plots
fig1 = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=[
        "<b>User Features with Low Variance</b>",
        "<b>User Features with High Variance</b>",
        "<b>Campaign Features</b>",
    ],
    specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
    horizontal_spacing=0.0001,
)
traces_low = px.scatter_3d(
    user_df_low.astype({"cohort": str}),
    x="user_f0",
    y="user_f1",
    z="user_fh",
    color="cohort",
).data
traces_high = px.scatter_3d(
    user_df_high.astype({"cohort": str}),
    x="user_f0",
    y="user_f1",
    z="user_fh",
    color="cohort",
).data
traces_camp = px.scatter_3d(
    camp_df,
    x="camp_f0",
    y="camp_f1",
    z="camp_fh",
).data
for tr in traces_low:
    fig1.add_trace(tr, row=1, col=1)
for tr in traces_high:
    fig1.add_trace(tr, row=1, col=2)
for tr in traces_camp:
    fig1.add_trace(tr, row=1, col=3)
fig1.update_traces(marker_size=3)
fig1.update_layout(
    height=500,
    width=1200,
    autosize=False,
    showlegend=False,
    margin={"l": 5, "r": 5, "t": 20, "b": 5},
    scene={
        "camera": {"eye": {"x": 1.9, "y": 1.9, "z": 1.7}},
        "xaxis_title": "Visible Feature 1",
        "yaxis_title": "Visible Feature 2",
        "zaxis_title": "Hidden Feature",
    },
    scene2={
        "camera": {"eye": {"x": 1.9, "y": 1.9, "z": 1.7}},
        "xaxis_title": "Visible Feature 1",
        "yaxis_title": "Visible Feature 2",
        "zaxis_title": "Hidden Feature",
    },
    scene3={
        "camera": {"eye": {"x": 1.8, "y": 1.8, "z": 1.8}},
        "xaxis_title": "Feature 1",
        "yaxis_title": "Feature 2",
        "zaxis_title": "Feature 3",
    },
    **PARAM_FONT_SZ,
)
fig1.write_image(os.path.join(fig_path, "fig1.svg"), scale=5)
fig1.write_image(os.path.join(fig_path, "fig1.png"), scale=5)


#%% fig2
# defs
fig_path = os.path.join(FIG_PATH, "fig2")
os.makedirs(fig_path, exist_ok=True)
options_all = {"opacity": 1, "showscale": False, "showlegend": True}
options = {
    "visible features": {
        "colorscale": "blues_r",
        "name": "Visible Features",
    },
    "all features": {
        "colorscale": "greens_r",
        "name": "All Features",
    },
    "real cohort id": {
        "colorscale": "reds_r",
        "name": "Cohort ID<br>+ Visible Features",
    },
}
scene_opts = {
    "aspectmode": "cube",
    "xaxis_title": "Hidden Features Variance",
    "yaxis_title": "Visible Feature Variance",
    "zaxis_title": "CV score",
    "camera": {"eye": {"x": 1.9, "y": 1.9, "z": 0}},
}
var_f_slice = 0.5
# load data
result = pd.read_csv(IN_RESULT).sort_values("feats", key=feat_order, ascending=False)
result_agg = result.groupby(["var_f", "var_fh", "feats", "split_by"]).agg(
    {"scores_test": ["mean", "sem", "median"]}
)
result_agg.columns = result_agg.columns.droplevel(0)
result_agg = result_agg.reset_index()
mi_df = (
    pd.read_csv(IN_MI)
    .groupby(["variable", "var_f", "var_fh"])["mi"]
    .median()
    .reset_index()
)
# make plot
# fig2a
fig2a = go.Figure()
for feat, f_df in result_agg.groupby("feats"):
    arr = f_df.set_index(["var_f", "var_fh"])["median"].to_xarray()
    try:
        opts = options_all | options[feat]
    except KeyError:
        continue
    trace = go.Surface(
        x=arr.coords["var_f"], y=arr.coords["var_fh"], z=arr.values, **opts
    )
    fig2a.add_trace(trace)
fig2a.update_layout(
    title="<b>Model Performance Using Cohort ID</b>",
    height=500,
    width=600,
    autosize=False,
    scene=scene_opts,
    margin={"l": 3, "r": 3, "t": 40, "b": 3},
    legend={"yanchor": "top", "y": 0.5},
    **PARAM_FONT_SZ,
)
fig2a.write_image(os.path.join(fig_path, "fig2a.svg"), scale=5)
fig2a.write_image(os.path.join(fig_path, "fig2a.png"), scale=5)
# fig2b
fig2b = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=[
        "<b>Performace of different models<b>",
        "<b>Mutual Information Shared with<br>Hidden User Feature<b>",
    ],
    vertical_spacing=0.25,
)
result_sub = result[
    (result["var_fh"] == PARAM_VAR_SLC)
    & result["feats"].isin(["visible features", "real cohort id", "all features"])
]
mi_sub = mi_df[
    (mi_df["var_fh"] == PARAM_VAR_SLC)
    & mi_df["variable"].isin(["user_f0", "user_f1", "real_cohort"])
]
for feat, f_df in result_sub.groupby("feats"):
    trace = go.Box(
        x=f_df["var_f"],
        y=f_df["scores_test"],
        name=PARAM_NMAP_FEAT[feat],
        marker={"color": PARAM_CMAP_FEAT[feat], "size": 3},
        line={"width": 1},
        legendgroup="1",
    )
    fig2b.add_trace(trace, row=1, col=1)
for feat, m_df in mi_sub.groupby("variable"):
    trace = go.Bar(
        x=m_df["var_f"],
        y=m_df["mi"],
        name=PARAM_NMAP_MI[feat],
        marker={"color": PARAM_CMAP_MI[feat]},
        legendgroup="2",
    )
    fig2b.add_trace(trace, row=2, col=1)
fig2b.update_xaxes(title="Visible Feature Variance")
fig2b.update_yaxes(title="CV Score", row=1, col=1)
fig2b.update_yaxes(title="Mutual Information", row=2, col=1)
fig2b.update_layout(
    margin={"l": 3, "r": 3, "b": 3, "t": 25},
    autosize=False,
    height=500,
    width=800,
    boxmode="group",
    barmode="stack",
    legend_tracegroupgap=220,
    **PARAM_FONT_SZ,
)
fig2b.write_image(os.path.join(fig_path, "fig2b.svg"), scale=5)
fig2b.write_image(os.path.join(fig_path, "fig2b.png"), scale=5)

#%% fig3
# defs
fig_path = os.path.join(FIG_PATH, "fig3")
os.makedirs(fig_path, exist_ok=True)
param_data = PARAM_DATA.copy()
param_data["num_cohort"] = 2
param_data["num_campaigns"] = 1
param_data["cohort_means"] = np.array([[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]])
param_data["cohort_variances"] = np.array([0.6, 0.6, 0.4])
param_model = {
    "n_cohort": 2,
    "feats": ["cohort", "user_f0", "user_f1", "camp_f0", "camp_f1", "camp_fh"],
    "cohort_feats": ["user_f0", "user_f1"],
    "user_feats": ["user_f0", "user_f1"],
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
user_scene_opts = {
    "aspectmode": "cube",
    "xaxis_title": "Visible Feature 1",
    "yaxis_title": "Visible Feature 2",
    "zaxis_title": "Hidden Feature",
    "camera": {"eye": {"x": 1.9, "y": -1.9, "z": 1}},
}
cmap_cluster = {0: qualitative.Plotly[0], 1: qualitative.Plotly[1]}
options_all = {"opacity": 1, "showscale": False, "showlegend": True}
options = {
    "visible features": {
        "colorscale": "blues_r",
        "name": "Visible Features",
    },
    "all features": {
        "colorscale": "greens_r",
        "name": "All Features",
    },
    "response-clustered cohort id": {
        "colorscale": "purples_r",
        "name": "Predicted Cohort ID<br>+ Visible Features",
    },
}
scene_opts = {
    "aspectmode": "cube",
    "xaxis_title": "Hidden Features Variance",
    "yaxis_title": "Visible Feature Variance",
    "zaxis_title": "CV score",
    "camera": {"eye": {"x": 1.9, "y": 1.9, "z": 0}},
}
# load data
result = pd.read_csv(IN_RESULT).sort_values("feats", key=feat_order, ascending=False)
result_agg = result.groupby(["var_f", "var_fh", "feats", "split_by"]).agg(
    {"scores_test": ["mean", "sem", "median"]}
)
result_agg.columns = result_agg.columns.droplevel(0)
result_agg = result_agg.reset_index()
mi_df = (
    pd.read_csv(IN_MI)
    .groupby(["variable", "var_f", "var_fh"])["mi"]
    .median()
    .reset_index()
)
data_df, user_df, _ = generate_data(camp_df=camp_df, **param_data)
model = CohortXGB(use_cohort_resp=False, **param_model)
model.fit(data_df)
user_df["cohort_pred"] = model.predict_cohort(user_df)
model = CohortXGB(use_cohort_resp=True, **param_model)
model.fit(data_df)
user_df["cohort_pred_resp"] = model.predict_cohort(user_df)
# make plot
# fig3a
fig3a = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=[
        "<b>Real Cohort ID</b>",
        "<b>Cohort ID Predicted with<br>Visible Features</b>",
        "<b>Cohort ID Predicted with<br>Visible Features and Responses</b>",
    ],
    specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
    horizontal_spacing=0.01,
)
tr_real = px.scatter_3d(
    user_df, x="user_f0", y="user_f1", z="user_fh", color="cohort"
).data
tr_coh = px.scatter_3d(
    user_df, x="user_f0", y="user_f1", z="user_fh", color="cohort_pred"
).data
tr_coh_re = px.scatter_3d(
    user_df, x="user_f0", y="user_f1", z="user_fh", color="cohort_pred_resp"
).data
for tr in tr_real:
    fig3a.add_trace(tr, row=1, col=1)
for tr in tr_coh:
    fig3a.add_trace(tr, row=1, col=2)
for tr in tr_coh_re:
    fig3a.add_trace(tr, row=1, col=3)
fig3a.update_traces(marker_size=3)
fig3a.update_xaxes(title_standoff=0)
fig3a.update_layout(
    autosize=False,
    height=500,
    width=1200,
    margin={"t": 60, "l": 3, "r": 3, "b": 3},
    showlegend=False,
    scene=user_scene_opts,
    scene2=user_scene_opts,
    scene3=user_scene_opts,
    **PARAM_FONT_SZ,
)
fig3a.write_image(os.path.join(fig_path, "fig3a.svg"), scale=5)
fig3a.write_image(os.path.join(fig_path, "fig3a.png"), scale=5)
# fig3b
fig3b = go.Figure()
for feat, f_df in result_agg.groupby("feats"):
    arr = f_df.set_index(["var_f", "var_fh"])["median"].to_xarray()
    try:
        opts = options_all | options[feat]
    except KeyError:
        continue
    trace = go.Surface(
        x=arr.coords["var_f"], y=arr.coords["var_fh"], z=arr.values, **opts
    )
    fig3b.add_trace(trace)
fig3b.update_layout(
    title="<b>Model Performance Using Predicted Cohort ID</b>",
    height=500,
    width=600,
    autosize=False,
    scene=scene_opts,
    margin={"l": 3, "r": 3, "t": 40, "b": 3},
    legend={"yanchor": "top", "y": 0.5},
    **PARAM_FONT_SZ,
)
fig3b.write_image(os.path.join(fig_path, "fig3b.svg"), scale=5)
fig3b.write_image(os.path.join(fig_path, "fig3b.png"), scale=5)
# fig3c
fig3c = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=[
        "<b>Performace of different models<b>",
        "<b>Mutual Information Shared with<br>Hidden User Feature<b>",
    ],
    vertical_spacing=0.25,
)
result_sub = result[
    (result["var_fh"] == PARAM_VAR_SLC)
    & result["feats"].isin(
        [
            "visible features",
            "real cohort id",
            "response-clustered cohort id",
            "all features",
        ]
    )
]
mi_sub = mi_df[
    (mi_df["var_fh"] == PARAM_VAR_SLC)
    & mi_df["variable"].isin(["user_f0", "user_f1", "real_cohort", "prd_cohort"])
]
for feat, f_df in result_sub.groupby("feats"):
    trace = go.Box(
        x=f_df["var_f"],
        y=f_df["scores_test"],
        name=PARAM_NMAP_FEAT[feat],
        marker={"color": PARAM_CMAP_FEAT[feat], "size": 3},
        line={"width": 1},
        legendgroup="1",
    )
    fig3c.add_trace(trace, row=1, col=1)
for feat, m_df in mi_sub.groupby("variable"):
    trace = go.Bar(
        x=m_df["var_f"],
        y=m_df["mi"],
        name=PARAM_NMAP_MI[feat],
        marker={"color": PARAM_CMAP_MI[feat]},
        legendgroup="2",
    )
    fig3c.add_trace(trace, row=2, col=1)
fig3c.update_xaxes(title="Visible Feature Variance")
fig3c.update_yaxes(title="CV Score", row=1, col=1)
fig3c.update_yaxes(title="Mutual Information", row=2, col=1)
fig3c.update_layout(
    margin={"l": 3, "r": 3, "b": 3, "t": 25},
    autosize=False,
    height=500,
    width=800,
    boxmode="group",
    barmode="stack",
    legend_tracegroupgap=200,
    **PARAM_FONT_SZ,
)
fig3c.write_image(os.path.join(fig_path, "fig3c.svg"), scale=5)
fig3c.write_image(os.path.join(fig_path, "fig3c.png"), scale=5)
