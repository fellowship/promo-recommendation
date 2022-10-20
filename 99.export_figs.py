#%% imports and definition
import os

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from routine.data_generation import generate_data

FIG_PATH = "./figs/export"
PARAM_DATA = {
    "num_users": 1000,
    "num_campaigns": 100,
    "samples_per_campaign": 1000,
    "num_cohort": 10,
    "fh_cohort": True,
    "response_sig_a": 10,
    "even_cohort": True,
    "cross_weight": None,
    "magnify_hf": 2,
    "perfect_camp": True,
}
PARAM_FONT_SZ = {"font_size": 10, "title_font_size": 24, "legend_title_font_size": 24}

np.random.seed(42)


def agg_freq(df):
    return (df["response"] == 1).sum() / len(df)


#%% fig 1
# defs
fig_path = os.path.join(FIG_PATH, "fig1")
os.makedirs(fig_path, exist_ok=True)
fig1 = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=[
        "<b>User Features with Low Variance</b>",
        "<b>User Features with High Variance</b>",
        "<b>Campaign Features</b>",
    ],
    specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
    horizontal_spacing=0.001,
)
_, user_df_low, camp_df = generate_data(cohort_variances=0.1, **PARAM_DATA)
_, user_df_high, camp_df = generate_data(cohort_variances=0.3, **PARAM_DATA)
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
        "camera": {"eye": {"x": 1.6, "y": 1.6, "z": 1.6}},
        "xaxis_title": "Visible Feature 1",
        "yaxis_title": "Visible Feature 2",
        "zaxis_title": "Hidden Feature",
    },
    scene2={
        "camera": {"eye": {"x": 1.6, "y": 1.6, "z": 1.6}},
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
    **PARAM_FONT_SZ
)
fig1.write_image(os.path.join(fig_path, "fig1.svg"), scale=5)
