#%% imports and definitions
import os

import numpy as np
import plotly.express as px

from routine.data_generation import generate_data

FIG_PATH = "./figs/data"
PARAM_FONT_SZ = {"font_size": 16, "title_font_size": 24, "legend_title_font_size": 24}
os.makedirs(FIG_PATH, exist_ok=True)

#%% generate data with hidden feature depending on cohort
# generate data
obs_df, user_df, camp_df = generate_data(
    num_users=1000,
    num_campaigns=100,
    samples_per_campaign=100000,
    num_cohort=10,
    cohort_variances=np.linspace(0.01, 0.1, 10),
    fh_cohort=True,
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
fig_user.write_html(os.path.join(FIG_PATH, "user_fh_cohort.html"))
# plot response cdf
resp_df = obs_df.groupby(["cohort", "response"]).size().rename("freq").reset_index()
fig_resp = px.bar(resp_df, x="response", y="freq", facet_col="cohort", facet_col_wrap=5)
fig_resp.add_hline(100000 * 100 / 20, line_dash="dot", line_color="gray")
fig_resp.update_layout(title="Hidden features dependent on cohorts", **PARAM_FONT_SZ)
fig_resp.write_html(os.path.join(FIG_PATH, "resp_fh_cohort.html"))

#%% generate data with hidden feature independnt of cohort
# generate data
obs_df, user_df, camp_df = generate_data(
    num_users=1000,
    num_campaigns=100,
    samples_per_campaign=100000,
    num_cohort=10,
    cohort_variances=0.05,
    fh_cohort=False,
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
    title="Hidden features independent of cohorts",
    **PARAM_FONT_SZ,
)
fig_user.write_html(os.path.join(FIG_PATH, "user_fh_ind.html"))
# plot campaign features
fig_camp = px.scatter_3d(camp_df, x="camp_f0", y="camp_f1", z="camp_fh")
fig_camp.update_traces(marker_size=3)
fig_camp.update_layout(legend={"itemsizing": "constant"}, **PARAM_FONT_SZ)
fig_camp.write_html(os.path.join(FIG_PATH, "camp.html"))
# plot response cdf
resp_df = obs_df.groupby(["cohort", "response"]).size().rename("freq").reset_index()
fig_resp = px.bar(resp_df, x="response", y="freq", facet_col="cohort", facet_col_wrap=5)
fig_resp.add_hline(100000 * 100 / 20, line_dash="dot", line_color="gray")
fig_resp.update_layout(title="Hidden features independent of cohorts", **PARAM_FONT_SZ)
fig_resp.write_html(os.path.join(FIG_PATH, "resp_fh_ind.html"))
