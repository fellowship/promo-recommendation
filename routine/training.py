def split_by_camp(df, prop):
    camp_splt = df.sort_values("camp_id").iloc[: int(len(df) * prop)]["camp_id"].max()
    return df[df["camp_id"] < camp_splt].copy(), df[df["camp_id"] >= camp_splt].copy()
