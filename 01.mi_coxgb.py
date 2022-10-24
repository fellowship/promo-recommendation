#%% imports and defs
import itertools as itt
import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from routine.data_generation import generate_data
from routine.lnc import MI
from routine.models import CohortXGB

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
PARAM_VAR_FH = np.linspace(0.1, 0.9, 9)
PARAM_VAR_F = np.linspace(0.1, 0.9, 9)
PARAM_NREPEAT = 30
PARAM_MODEL = {
    "feats": ["cohort", "user_f0", "user_f1", "camp_f0", "camp_f1", "camp_fh"],
    "cohort_feats": ["user_f0", "user_f1"],
    "user_feats": ["user_f0", "user_f1"],
    "use_cohort_resp": True,
    "max_depth": 5,
    "learning_rate": 1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "use_label_encoder": False,
}
OUT_PATH = "./intermediate/mi_coxgb"

os.makedirs(OUT_PATH, exist_ok=True)
np.random.seed(42)

#%% generate mi
mi_df = []
for v_f, v_fh, irpt in tqdm(
    list(itt.product(PARAM_VAR_F, PARAM_VAR_FH, range(PARAM_NREPEAT)))
):
    cohort_var = np.array([v_f, v_f, v_fh])
    cur_data, user_df, _ = generate_data(cohort_variances=cohort_var, **PARAM_DATA)
    model = CohortXGB(n_cohort=PARAM_DATA["num_cohort"], **PARAM_MODEL)
    model.fit(cur_data.copy())
    cohort_prd = model.predict_cohort(user_df)
    mi = pd.DataFrame(
        {
            "variable": ["user_f0", "user_f1", "real_cohort", "prd_cohort"],
            "mi": [
                MI.mi_LNC(user_df[["user_f0", "user_fh"]].values.T),
                MI.mi_LNC(user_df[["user_f1", "user_fh"]].values.T),
                MI.mi_LNC(user_df[["cohort", "user_fh"]].values.T),
                MI.mi_LNC(
                    np.stack(
                        [np.array(cohort_prd), np.array(user_df["user_fh"])], axis=0
                    )
                ),
            ],
            "var_f": v_f,
            "var_fh": v_fh,
            "irpt": irpt,
        }
    )
    mi_df.append(mi)
mi_df = pd.concat(mi_df, ignore_index=True)
mi_df.to_csv(os.path.join(OUT_PATH, "mi_df.csv"), index=False)
