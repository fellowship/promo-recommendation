#%% imports and definitions
import itertools as itt
import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from routine.data_generation import generate_data
from routine.lnc import MI
from routine.plotting import imshow

FIG_PATH = "./figs/feat_mi"
PARAM_DATA = {
    "num_users": 1000,
    "num_campaigns": 100,
    "samples_per_campaign": 100,
    "num_cohort": 10,
    "even_cohort": True,
    "response_sig_a": 10,
    "cross_weight": None,
    "magnify_hf": 1,
}
PARAM_FONT_SZ = {"font_size": 16, "title_font_size": 24, "legend_title_font_size": 24}
PARAM_VAR = np.around(np.linspace(0.05, 0.6, 12), 2)
OUT_PATH = "./intermediate/feat_mi"
os.makedirs(FIG_PATH, exist_ok=True)
os.makedirs(OUT_PATH, exist_ok=True)

#%% generate data
mi_df = []
for cvar in tqdm(PARAM_VAR):
    obs_df, user_df, camp_df = generate_data(cohort_variances=cvar, **PARAM_DATA)
    for featA, featB in itt.combinations_with_replacement(
        ["cohort", "user_f0", "user_f1", "user_fh"], 2
    ):
        mi = MI.mi_LNC(user_df[[featA, featB]].values.T)
        mi_df.append(
            pd.Series({"featA": featA, "featB": featB, "mi": mi, "cohort_var": cvar})
            .to_frame()
            .T
        )
mi_df = pd.concat(mi_df, ignore_index=True)
mi_df.to_csv(os.path.join(OUT_PATH, "mi_df.csv"))
#%% plot mi
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
fig.write_html(os.path.join(FIG_PATH, "feat_mi.html"))
