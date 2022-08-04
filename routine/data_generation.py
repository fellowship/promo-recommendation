import numpy as np
import pandas as pd

from .utilities import sigmoid, unit_norm


def fibonacci_sphere(n_samples=1000):
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians
    for i in range(n_samples):
        y = 1 - (i / float(n_samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append((x, y, z))
    return np.array(points)


def divide_circle(n_samples=1000):
    angs = 2 * np.pi / n_samples * np.arange(n_samples)
    return np.stack([np.cos(angs), np.sin(angs)], axis=1)


def sample_cohort(cohort, variances, n_features):
    cohort = np.array(cohort)
    n_cohort = len(np.unique(cohort))
    if np.isscalar(variances):
        variances = np.repeat(variances, n_cohort)
    if n_features == 1:
        means = np.linspace(-1, 1, n_cohort)[:, np.newaxis]
    elif n_features == 2:
        means = divide_circle(n_cohort)
    elif n_features == 3:
        means = fibonacci_sphere(n_cohort)
    else:
        raise NotImplementedError("Can only generate up to 3d features")
    means = means[cohort, :]
    variances = variances[cohort, np.newaxis]
    return np.random.normal(loc=means, scale=variances)


def get_sample_freq(n_samples, nunique):
    freq = np.random.random(nunique)
    freq = np.around(freq / freq.sum() * n_samples).astype(int).clip(1, None)
    diff = n_samples - freq.sum()
    if diff != 0:
        freq = np.sort(freq)
        n_modify = np.abs(diff)
        freq[-n_modify:] = freq[-n_modify:] + np.sign(diff)
    assert freq.sum() == n_samples
    np.random.shuffle(freq)
    return freq


def generate_data(
    num_users,
    num_campaigns,
    samples_per_campaign,
    num_cohort,
    cohort_variances,
    fh_cohort=True,
    response_sig_a=10,
):
    # get number of samples
    nsample = num_campaigns * samples_per_campaign
    # assign users to cohorts uniformly with random frequency
    uids = np.arange(num_users)
    user_df = pd.DataFrame(
        {
            "user_id": uids,
            "cohort": np.repeat(
                np.arange(num_cohort),
                [len(c) for c in np.array_split(uids, num_cohort)],
            ),
            "freq": get_sample_freq(nsample, num_users),
        }
    )
    # generate feature vector for each user
    if fh_cohort:
        feats = unit_norm(sample_cohort(user_df["cohort"], cohort_variances, 3))
        user_df = user_df.assign(
            **{"user_f0": feats[:, 0], "user_f1": feats[:, 1], "user_fh": feats[:, 2]}
        )
    else:
        feats = sample_cohort(user_df["cohort"], cohort_variances, 2)
        fh = sample_cohort(user_df["cohort"], cohort_variances, 1)
        np.random.shuffle(fh)
        feats = unit_norm(np.concatenate([feats, fh], axis=1))
        user_df = user_df.assign(
            **{"user_f0": feats[:, 0], "user_f1": feats[:, 1], "user_fh": feats[:, 2]}
        )
    # generate campaigns with random frequency and uniform features
    camp_df = pd.DataFrame(
        {
            "camp_id": np.arange(num_campaigns),
            "freq": get_sample_freq(nsample, num_campaigns),
        }
    )
    feats = fibonacci_sphere(num_campaigns)
    camp_df = camp_df.assign(
        **{"camp_f0": feats[:, 0], "camp_f1": feats[:, 1], "camp_fh": feats[:, 2]}
    )
    # build observations
    user_ids = np.repeat(user_df["user_id"].values, user_df["freq"].values)
    camp_ids = np.repeat(camp_df["camp_id"].values, camp_df["freq"].values)
    np.random.shuffle(user_ids)
    np.random.shuffle(camp_ids)
    obs_df = pd.DataFrame({"user_id": user_ids, "camp_id": camp_ids})
    obs_df = obs_df.merge(user_df.drop(columns="freq"), how="left", on="user_id")
    obs_df = obs_df.merge(camp_df.drop(columns="freq"), how="left", on="camp_id")
    obs_df["iprod"] = (
        obs_df[["user_f0", "user_f1", "user_fh"]].values
        * obs_df[["camp_f0", "camp_f1", "camp_fh"]].values
    ).sum(axis=1)
    obs_df["response"] = sigmoid(
        obs_df["iprod"],
        a=response_sig_a,
        b=-0.5,
    )
    return obs_df, user_df, camp_df
