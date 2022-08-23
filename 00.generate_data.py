#%% imports and definitions
import os
import random
import sys

import numpy as np
import plotly.express as px
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

import routine.data_generation

FIG_PATH = "./figs/data"
PARAM_FONT_SZ = {"font_size": 16, "title_font_size": 24, "legend_title_font_size": 24}
os.makedirs(FIG_PATH, exist_ok=True)


def agg_freq(df):
    return (df["response"] == 1).sum() / len(df)


#%% generate data
from routine.data_generation import generate_data
obs_df, user_df, camp_df = generate_data(
    num_users=1000,
    num_campaigns=100,
    samples_per_campaign=10000,
    num_cohort=10,
    cohort_variances=0.2,
    fh_cohort=True,
    response_sig_a=10,
    even_cohort=True,
    cross_response=False,
    magnify_hf=1,
    var_fac_hf=1,
    kmeans=False
)
# plot user features
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
fig_user.write_html(os.path.join(FIG_PATH, "user.html"))
# plot campaign features
fig_camp = px.scatter_3d(camp_df, x="camp_f0", y="camp_f1", z="camp_fh")
fig_camp.update_traces(marker_size=3)
fig_camp.update_layout(legend={"itemsizing": "constant"}, **PARAM_FONT_SZ)
fig_camp.write_html(os.path.join(FIG_PATH, "camp.html"))

# plot response cdf
resp_df = (
    obs_df.groupby(["camp_id", "cohort"]).apply(agg_freq).rename("freq").reset_index()
)

#%% write all features to dataframe
fig_resp = px.bar(resp_df, x="cohort", y="freq", facet_col="camp_id", facet_col_wrap=10)
fig_resp.add_hline(0.5, line_dash="dot", line_color="gray")
fig_resp.update_layout(title="Hidden features dependent on cohorts", **PARAM_FONT_SZ)
fig_resp.write_html(os.path.join(FIG_PATH, "resp.html"))

# Save the data frame
obs_df.to_csv("observation.csv")
