#%% imports and definitions
import itertools as itt
import os

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.express.colors import qualitative
from tqdm.auto import tqdm

from routine.clustering import Kmeans_cluster
from routine.data_generation import generate_data
from routine.lnc import MI
from routine.plotting import imshow
from routine.models import response_cluster

FIG_PATH = "./figs/feat_mi"
PARAM_DATA = {
    "num_users": 1000,
    "num_campaigns": 50,
    "samples_per_campaign": 1000,
    "num_cohort": 10,
    "even_cohort": True,
    "response_sig_a": 10,
    "cross_weight": None,
    "magnify_hf": 2,
    "learning_rate_story": True
}
PARAM_FONT_SZ = {"font_size": 16, "title_font_size": 24, "legend_title_font_size": 24}
PARAM_VAR = np.around(np.linspace(0.1, 1, 10), 2)
PARAM_NREPEAT = 30
OUT_PATH = "./intermediate/feat_mi"
os.makedirs(FIG_PATH, exist_ok=True)
os.makedirs(OUT_PATH, exist_ok=True)

#%% generate data with uniform variance
mi_df = []
for cvar in tqdm(PARAM_VAR):
    for irpt in range(PARAM_NREPEAT):
        obs_df, user_df, camp_df = generate_data(cohort_variances=cvar, **PARAM_DATA)
        for featA, featB in itt.combinations_with_replacement(
            ["cohort", "user_f0", "user_f1", "user_fh"], 2
        ):
            mi = MI.mi_LNC(user_df[[featA, featB]].values.T)
            mi_df.append(
                pd.Series(
                    {
                        "featA": featA,
                        "featB": featB,
                        "mi": mi,
                        "cohort_var": cvar,
                        "irpt": irpt,
                    }
                )
                .to_frame()
                .T
            )
mi_df = pd.concat(mi_df, ignore_index=True)
mi_df.to_csv(os.path.join(OUT_PATH, "mi_df_uniform.csv"), index=False)
#%% plot mi matrix
mi_df = pd.read_csv(os.path.join(OUT_PATH, "mi_df_uniform.csv"))
mi_df = mi_df.groupby(["featA", "featB", "cohort_var"])["mi"].mean().reset_index()
fig = imshow(
    mi_df,
    facet_row=None,
    facet_col="cohort_var",
    subplot_args={"col_wrap": 6, "horizontal_spacing": 0.05},
    x="featA",
    y="featB",
    z="mi",
    coloraxis="coloraxis",
    text="mi",
    texttemplate="%{text:.1f}",
)
fig.update_layout(
    {"coloraxis": {"cmin": 0, "cmax": 3, "colorscale": "viridis"}}, **PARAM_FONT_SZ
)
fig.write_html(os.path.join(FIG_PATH, "mi_matrix.html"))

#%% plot fh mi bar
mi_df = pd.read_csv(os.path.join(OUT_PATH, "mi_df_uniform.csv"))
mi_df = (
    mi_df[(mi_df["featB"] == "user_fh") & (mi_df["featA"] != "user_fh")]
    .groupby(["featA", "featB", "cohort_var"])["mi"]
    .mean()
    .reset_index()
)
fig = px.bar(mi_df, x="cohort_var", y="mi", color="featA")
fig.update_layout(
    legend_title="Variable",
    xaxis_title="Cohort Variance",
    yaxis_title="Mutual Information<br>Shared with Hidden Feature",
    **PARAM_FONT_SZ
)
fig.write_html(os.path.join(FIG_PATH, "mi_fh_bar.html"))
fig.write_image(os.path.join(FIG_PATH, "mi_fh_bar.png"), width=1500, height=800, scale=1)

#%% generate data with clustering
mi_df = []
for cvar in tqdm(PARAM_VAR):
    for irpt in range(PARAM_NREPEAT):
        obs_df, user_df, camp_df = generate_data(
            cohort_variances=np.array([cvar, cvar, 0.2]), **PARAM_DATA
        )
        user_df = user_df.rename(columns={"cohort": "cohort_real"})
        user_df["cohort_cluster"] = Kmeans_cluster(
            user_df[["user_f0", "user_f1"]], PARAM_DATA["num_cohort"]
        )
        for featA, featB in itt.combinations_with_replacement(
            ["cohort_real", "cohort_cluster", "user_f0", "user_f1", "user_fh"], 2
        ):
            mi = MI.mi_LNC(user_df[[featA, featB]].values.T)
            mi_df.append(
                pd.Series(
                    {
                        "featA": featA,
                        "featB": featB,
                        "mi": mi,
                        "cohort_var": cvar,
                        "irpt": irpt,
                    }
                )
                .to_frame()
                .T
            )
mi_df = pd.concat(mi_df, ignore_index=True)
mi_df.to_csv(os.path.join(OUT_PATH, "mi_df_cluster.csv"), index=False)

#%% plot fh mi bar
mi_df = pd.read_csv(os.path.join(OUT_PATH, "mi_df_cluster.csv"))
mi_df = (
    mi_df[(mi_df["featB"] == "user_fh") & (mi_df["featA"] != "user_fh")]
    .groupby(["featA", "featB", "cohort_var"])["mi"]
    .mean()
    .reset_index()
)
fig = px.bar(
    mi_df,
    x="cohort_var",
    y="mi",
    color="featA",
    color_discrete_map={
        "user_f0": qualitative.Plotly[2],
        "cohort_real": qualitative.Plotly[1],
        "cohort_cluster": qualitative.Plotly[3],
        "user_f1": qualitative.Plotly[0],
    },
)
fig.update_layout(
    legend_title="Variable",
    xaxis_title="Visible Feature Variance",
    yaxis_title="Mutual Information<br>Shared with Hidden Feature",
    **PARAM_FONT_SZ
)
fig.write_html(os.path.join(FIG_PATH, "mi_fh_bar_cluster.html"))
fig.write_image(os.path.join(FIG_PATH, "mi_fh_bar_cluster.png"), width=1500, height=800, scale=1)

#%% generate data with response clustering and k means clustering
cohort_feats = ["user_f0", "user_f1"]
mi_df = []
for cvar in tqdm(PARAM_VAR):
    for irpt in range(PARAM_NREPEAT):
        obs_df, user_df, camp_df = generate_data(
            cohort_variances=np.array([cvar, cvar, 0.1]), **PARAM_DATA
        )

        user_df = user_df.rename(columns={"cohort": "cohort_real"})
        user_df["cohort_cluster"] = Kmeans_cluster(
            user_df[["user_f0", "user_f1"]], PARAM_DATA["num_cohort"]
        )
        resp_df = response_cluster(obs_df, cohort_feats, n_cohort=PARAM_DATA["num_cohort"])
        user_df = user_df.merge(resp_df, how="left", on="user_id")

        for featA, featB in itt.combinations_with_replacement(
            ["cohort_real", "cohort_cluster", "cohort_response_cluster", "user_f0", "user_f1", "user_fh"], 2
        ):
            mi = MI.mi_LNC(user_df[[featA, featB]].values.T)
            mi_df.append(
                pd.Series(
                    {
                        "featA": featA,
                        "featB": featB,
                        "mi": mi,
                        "cohort_var": cvar,
                        "irpt": irpt,
                    }
                )
                .to_frame()
                .T
            )
mi_df = pd.concat(mi_df, ignore_index=True)
mi_df.to_csv(os.path.join(OUT_PATH, "mi_df_response_cluster.csv"), index=False)

#%% plot fh mi bar
mi_df = pd.read_csv(os.path.join(OUT_PATH, "mi_df_response_cluster.csv"))
mi_df = (
    mi_df[(mi_df["featB"] == "user_fh") & (mi_df["featA"] != "user_fh")]
    .groupby(["featA", "featB", "cohort_var"])["mi"]
    .mean()
    .reset_index()
)
fig = px.bar(
    mi_df,
    x="cohort_var",
    y="mi",
    color="featA",
    color_discrete_map={
        "user_f0": qualitative.Plotly[2],
        "cohort_real": qualitative.Plotly[1],
        "cohort_cluster": qualitative.Plotly[3],
        "cohort_response_cluster": qualitative.Plotly[4],
        "user_f1": qualitative.Plotly[0],
    },
)
fig.update_layout(
    legend_title="Variable",
    xaxis_title="Visible Feature Variance",
    yaxis_title="Mutual Information<br>Shared with Hidden Feature",
    **PARAM_FONT_SZ
)
fig.write_html(os.path.join(FIG_PATH, "mi_fh_bar_response_cluster.html"))
fig.write_image(os.path.join(FIG_PATH, "mi_fh_bar_response_cluster.png"), width=1500, height=800, scale=1)
