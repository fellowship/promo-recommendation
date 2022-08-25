#%% imports and definitions
import os

import plotly.express as px

from routine.data_generation import generate_data
from routine.clustering import Kmeans_cluster

FIG_PATH = "./figs/data"
PARAM_FONT_SZ = {"font_size": 16, "title_font_size": 24, "legend_title_font_size": 24}
os.makedirs(FIG_PATH, exist_ok=True)


def agg_freq(df):
    return (df["response"] == 1).sum() / len(df)


#%% generate data
obs_df, user_df, camp_df = generate_data(
    num_users=1000,
    num_campaigns=100,
    samples_per_campaign=10000,
    num_cohort=10,
    cohort_variances=0.6,
    fh_cohort=True,
    response_sig_a=10,
    even_cohort=True,
    cross_weight=None,
    magnify_hf=1,
)
# kmean cluster of cohort
user_df["cohort_cluster"] = Kmeans_cluster(user_df[["user_f0", "user_f1"]], 10)
# plot user features colored by real cohort
fig_user = px.scatter_3d(
    user_df.astype({"cohort": str}),
    x="user_f0",
    y="user_f1",
    z="user_fh",
    color="cohort",
)
fig_user.update_traces(marker_size=3)
fig_user.update_layout(
    legend={"itemsizing": "constant"},
    title="Hidden features dependent on cohorts",
    **PARAM_FONT_SZ,
)
fig_user.write_html(os.path.join(FIG_PATH, "user_real_cohort.html"))
# plot user features colored by clustered cohort
fig_user = px.scatter_3d(
    user_df.astype({"cohort_cluster": str}),
    x="user_f0",
    y="user_f1",
    z="user_fh",
    color="cohort_cluster",
)
fig_user.update_traces(marker_size=3)
fig_user.update_layout(
    legend={"itemsizing": "constant"},
    title="Hidden features dependent on cohorts",
    **PARAM_FONT_SZ,
)
fig_user.write_html(os.path.join(FIG_PATH, "user_cluster_cohort.html"))
# plot campaign features
fig_camp = px.scatter_3d(camp_df, x="camp_f0", y="camp_f1", z="camp_fh")
fig_camp.update_traces(marker_size=3)
fig_camp.update_layout(legend={"itemsizing": "constant"}, **PARAM_FONT_SZ)
fig_camp.write_html(os.path.join(FIG_PATH, "camp.html"))
# plot response cdf
resp_df = (
    obs_df.groupby(["camp_id", "cohort"]).apply(agg_freq).rename("freq").reset_index()
)
fig_resp = px.bar(resp_df, x="cohort", y="freq", facet_col="camp_id", facet_col_wrap=10)
fig_resp.add_hline(0.5, line_dash="dot", line_color="gray")
fig_resp.update_layout(title="Hidden features dependent on cohorts", **PARAM_FONT_SZ)
fig_resp.write_html(os.path.join(FIG_PATH, "resp.html"))
