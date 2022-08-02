#%% imports and definitions
import os

import numpy as np
import plotly.express as px

from routine.data_generation import generate_data

FIG_PATH = "./figs/data"
os.makedirs(FIG_PATH, exist_ok=True)

#%% generate data
obs_df, user_df, camp_df = generate_data(
    num_users=1000,
    num_campaigns=100,
    samples_per_campaign=100000,
    num_cohort=10,
    cohort_variances=np.linspace(0.01, 0.1, 10),
    fh_cohort=True,
    response_sig_a=10,
)

#%% plot user features
fig_user = px.scatter_3d(
    user_df.astype({"cohort": str}),
    x="user_f0",
    y="user_f1",
    z="user_fh",
    color="cohort",
)
fig_user.update_traces(marker_size=2)
fig_user.update_layout(legend={"itemsizing": "constant"})
fig_user.write_html(os.path.join(FIG_PATH, "user.html"))

#%% plot campaign features
fig_camp = px.scatter_3d(camp_df, x="camp_f0", y="camp_f1", z="camp_fh")
fig_camp.update_traces(marker_size=2)
fig_camp.update_layout(legend={"itemsizing": "constant"})
fig_camp.write_html(os.path.join(FIG_PATH, "camp.html"))

#%% plot response cdf
fig_resp = px.histogram(
    obs_df, x="response", nbins=500, histnorm="probability", cumulative=True
)
fig_resp.write_image(os.path.join(FIG_PATH, "resp.svg"))
