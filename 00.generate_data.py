# %% imports and definitions
import os

import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import routine.models
from routine.data_generation import generate_data
from routine.clustering import Kmeans_cluster
from routine.models import AddPCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

FIG_PATH = "./figs/data"
PARAM_FONT_SZ = {"font_size": 16, "title_font_size": 24, "legend_title_font_size": 16}
os.makedirs(FIG_PATH, exist_ok=True)


def agg_freq(df):
    return (df["response"] == 1).sum() / len(df)


# %% generate data
obs_df, user_df, camp_df = generate_data(
    num_users=1000,
    num_campaigns=100,
    samples_per_campaign=10000,
    num_cohort=10,
    cohort_variances=0.05,
    fh_cohort=True,
    response_sig_a=10,
    even_cohort=True,
    cross_weight=None,
    magnify_hf=1,
    learning_rate_story=False
)
#%%
obs_df, _ = AddPCA(obs_df, 30)
obs_df = obs_df.sort_values(["camp_id", "user_id"])
resp_ls = np.column_stack([resp["response"] for resp in np.array_split(obs_df, 30)])
_, S, _ = np.linalg.svd(resp_ls)
df_SIG = pd.DataFrame({"Modes": np.arange(len(S)), "Percent weight": S / np.sum(S)})
fig_SIG = px.line(df_SIG, x="Modes", y="Percent weight", title='Singular value decay', markers=True)
fig_SIG.write_html(os.path.join(FIG_PATH, "fig_SIG.html"))

pca_df = obs_df[['cohort', 'user_f0', 'user_f1', 'PCA_1', 'PCA_2', 'PCA_3']].copy()
fig_PCA = px.scatter_3d(
    pca_df.astype({"cohort": str}),
    x="user_f0",
    y="user_f1",
    z="PCA_3",
    color="cohort",
)
fig_PCA.update_traces(marker_size=2)
fig_PCA.update_layout(
    legend={"itemsizing": "constant"},
    title="features and PCA component 3",
    **PARAM_FONT_SZ,
)
fig_PCA.write_html(os.path.join(FIG_PATH, "feat_pca_3.html"))

df_small = obs_df.head()
fig_df = make_subplots(rows=2, cols=1, specs=[[{"type": "table"}],
           [{"type": "scatter3D"}]], subplot_titles=("Modified dataframe", "PCA component 1"), row_heights=[0.3, 0.7])
fig_df.add_trace(go.Table(
    header=dict(values=list(df_small.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=df_small.transpose().values.round(2).tolist(),
               fill_color='lavender',
               align='left')
), row=1, col=1)
fig_df.add_trace(go.Scatter3d(
    x=obs_df["user_f0"],
    y=obs_df["user_f1"],
    z=obs_df["PCA_1"],
    mode="markers",
    marker=dict(
        size=2,
        color=obs_df["cohort"]
    )
), row=2, col=1)
fig_df.write_html(os.path.join(FIG_PATH, "fig_df.html"))

#%%
df_small = obs_df.head()
df_small.rename(columns={'camp_fh': 'camp_f2'}, inplace=True)
fig_df = go.Figure(data=[go.Table(
    header=dict(values=list(df_small.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=df_small.transpose().values.round(2).tolist(),
               fill_color='lavender',
               align='left'))
])
fig_df.write_html(os.path.join(FIG_PATH, "fig_df.html"))
fig_df.write_image(os.path.join(FIG_PATH, "fig_df.png"), width=800, height=400, scale=1)

# %%
# kmean cluster of cohort
user_df["cohort_cluster"] = Kmeans_cluster(user_df[["user_f0", "user_f1"]], 10)
# plot user features colored by real cohort
fig_user = px.scatter_3d(
    user_df.astype({"cohort": str}),
    x="user_f0",
    y="user_f1",
    z="user_fh",
    color="cohort",
    labels={
        "user_f0": 'user_f0',
        "user_f1": 'user_f1',
        "user_fh": 'user_fh'
    }
)
fig_user.update_traces(marker_size=1.5)
fig_user.update_layout(
    legend={"itemsizing": "constant"},
    # title="Hidden features dependent on cohorts",
    **PARAM_FONT_SZ,
)
fig_user.write_html(os.path.join(FIG_PATH, "user_real_cohort.html"))
fig_user.write_image(os.path.join(FIG_PATH, "user_real_cohort.png"), width=850, height=800, scale=1)

# plot user features colored by clustered cohort
fig_user = px.scatter_3d(
    user_df.astype({"cohort_cluster": str}),
    x="user_f0",
    y="user_f1",
    z="user_fh",
    color="cohort_cluster",
    labels={
        "user_f0": 'user_f0',
        "user_f1": 'user_f1',
        "user_fh": 'user_fh'
    }
)
fig_user.update_traces(marker_size=1.5)
fig_user.update_layout(
    legend={"itemsizing": "constant"},
    # title="Hidden features dependent on cohorts",
    **PARAM_FONT_SZ,
)
fig_user.write_html(os.path.join(FIG_PATH, "user_cluster_cohort.html"))
fig_user.write_image(os.path.join(FIG_PATH, "user_cluster_cohort.png"), width=900, height=800, scale=1)

# plot campaign features
fig_camp = px.scatter_3d(camp_df,
     x="camp_f0",
     y="camp_f1",
     z="camp_fh",
     labels={
         "camp_f0": 'camp_f0',
         "camp_f1": 'camp_f1',
         "camp_fh": 'camp_f2'
     }
)
fig_camp.update_traces(marker_size=1.5)
fig_camp.update_layout(legend={"itemsizing": "constant"}, **PARAM_FONT_SZ)
fig_camp.write_html(os.path.join(FIG_PATH, "camp.html"))
fig_camp.write_image(os.path.join(FIG_PATH, "camp.png"), width=850, height=800, scale=1)

# plot response cdf
resp_df = (
    obs_df.groupby(["camp_id", "cohort"]).apply(agg_freq).rename("freq").reset_index()
)
fig_resp = px.bar(resp_df, x="cohort", y="freq", facet_col="camp_id", facet_col_wrap=10)
fig_resp.add_hline(0.5, line_dash="dot", line_color="gray")
fig_resp.update_layout(title="Hidden features dependent on cohorts", **PARAM_FONT_SZ)
fig_resp.write_html(os.path.join(FIG_PATH, "resp.html"))
