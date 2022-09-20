import pandas as pd
import numpy as np


def split_by_id(df: pd.DataFrame, prop: float, id_col: str = "camp_id"):
    splt = df.sort_values(id_col).iloc[: int(len(df) * prop)][id_col].max()
    return df[df[id_col] < splt].copy(), df[df[id_col] >= splt].copy()


def cv_by_id(df: pd.DataFrame, fold: int, id_col: str = "camp_id"):
    ids = np.unique(df[id_col])
    for cur_test_id in np.array_split(ids, fold):
        cur_train_id = np.array(list(set(ids) - set(cur_test_id)))
        yield (
            df[df[id_col].isin(cur_train_id)].copy(),
            df[df[id_col].isin(cur_test_id)].copy(),
        )
