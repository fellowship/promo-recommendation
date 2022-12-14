import numpy as np
import pandas as pd
from sklearn.manifold import SpectralEmbedding

from .utilities import norm, sigmoid


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


def eigmap_eqdist(n_points, n_dim):
    eig = SpectralEmbedding(n_components=n_dim, affinity="precomputed", n_jobs=-1)
    crd = eig.fit_transform(np.ones((n_points, n_points)))
    for i in range(n_dim):
        crd[:, i] = (norm(crd[:, i]) - 0.5) * 2
    return crd


def sample_means(n_points, n_dim, method="auto"):
    if n_dim == 1:
        return np.linspace(-1, 1, n_points)[:, np.newaxis]
    if method == "auto":
        method = "eigmap" if n_points < 15 else "random"
    if method == "random":
        return (np.random.random((n_points, n_dim)) - 0.5) * 2
    elif method == "eigmap":
        return eigmap_eqdist(n_points, n_dim)


def sample_cohort(cohort, variances, n_features, means):
    cohort = np.array(cohort)
    n_cohort = len(np.unique(cohort))
    if means is not None:
        assert means.ndim == 2
        assert means.shape[0] == n_cohort
        assert means.shape[1] == n_features
    if np.isscalar(variances):
        variances = np.full((n_cohort, n_features), variances)
    elif variances.ndim == 1:
        if variances.shape[0] == n_cohort:
            variances = np.repeat(variances[:, np.newaxis], n_features, axis=1)
        elif variances.shape[0] == n_features:
            variances = np.repeat(variances[np.newaxis, :], n_cohort, axis=0)
        else:
            raise ValueError(
                "Cannot interpret variances. 1d array must has shape n_cohort or n_features"
            )
    else:
        assert (
            variances.ndim == 2
            and variances.shape[0] == n_cohort
            and variances.shape[1] == n_features
        ), "Cannot interpret variances, 2d array must has shape (n_cohort, n_features)"
    if means is None:
        means = sample_means(n_cohort, n_features)
    means = means[cohort, :]
    variances = variances[cohort, :]
    smps = np.random.normal(loc=means, scale=variances)
    for i in range(smps.shape[1]):
        smps[:, i] = (norm(smps[:, i]) - 0.5) * 2
    return smps, means


def get_sample_freq(n_samples, nunique, min_count=1):
    freq = np.random.random(nunique)
    freq = np.around(freq / freq.sum() * n_samples).astype(int).clip(min_count, None)
    while True:
        diff = n_samples - freq.sum()
        if diff == 0:
            break
        freq = np.sort(freq)
        n_modify = np.abs(diff)
        freq[-n_modify:] = (freq[-n_modify:] + np.sign(diff)).clip(min_count, None)
    np.random.shuffle(freq)
    return freq


def generate_data(
    num_users,
    num_campaigns,
    samples_per_campaign,
    num_cohort,
    cohort_variances,
    cohort_means=None,
    fh_cohort=True,
    response_sig_a=10,
    even_cohort=True,
    min_user_per_cohort=10,
    cross_weight=None,
    magnify_hf=1,
    user_df=None,
    camp_df=None,
    perfect_camp=False,
):
    # get number of samples
    nsample = num_campaigns * samples_per_campaign
    # generate user df
    if user_df is None:
        # assign users to cohorts uniformly with random frequency
        uids = np.arange(num_users)
        if even_cohort:
            cohort_freq = [len(c) for c in np.array_split(uids, num_cohort)]
        else:
            cohort_freq = get_sample_freq(num_users, num_cohort, min_user_per_cohort)
        user_df = pd.DataFrame(
            {
                "user_id": uids,
                "cohort": np.repeat(np.arange(num_cohort), cohort_freq),
                "freq": get_sample_freq(nsample, num_users),
            }
        )
        # generate feature vector for each user
        if fh_cohort:
            feats, means = sample_cohort(
                user_df["cohort"], cohort_variances, 3, cohort_means
            )
            user_df = user_df.assign(
                **{
                    "user_f0": feats[:, 0],
                    "user_f1": feats[:, 1],
                    "user_fh": magnify_hf * feats[:, 2],
                }
            )
        else:
            if cohort_variances.ndim == 1 and cohort_variances.shape[0] == 3:
                var_vis = cohort_variances[:2]
                var_fh = cohort_variances[2]
            else:
                var_vis, var_fh = cohort_variances, cohort_variances
            if cohort_means is not None:
                assert cohort_means.ndim == 2 and cohort_means.shape[1] == 3
                mean_vis, mean_fh = cohort_means[:, :2], cohort_means[:, :2]
            feats, _ = sample_cohort(user_df["cohort"], var_vis, 2, mean_vis)
            fh, _ = sample_cohort(user_df["cohort"], var_fh, 1, mean_fh)
            np.random.shuffle(fh)
            feats = np.concatenate([feats, fh], axis=1)
            user_df = user_df.assign(
                **{
                    "user_f0": feats[:, 0],
                    "user_f1": feats[:, 1],
                    "user_fh": magnify_hf * feats[:, 2],
                }
            )
    # generate campaigns with random frequency and uniform features
    if camp_df is None:
        camp_df = pd.DataFrame(
            {
                "camp_id": np.arange(num_campaigns),
                "freq": get_sample_freq(nsample, num_campaigns),
            }
        )
        feats = sample_means(num_campaigns, n_dim=3)
        camp_df = camp_df.assign(
            **{"camp_f0": feats[:, 0], "camp_f1": feats[:, 1], "camp_fh": feats[:, 2]}
        )
    # build observations
    if perfect_camp:
        user_ids = np.tile(np.array(user_df["user_id"]), num_campaigns)
        camp_ids = np.repeat(np.array(camp_df["camp_id"]), num_users)
        obs_df = (
            pd.DataFrame({"user_id": user_ids, "camp_id": camp_ids})
            .sample(frac=1)
            .reset_index(drop=True)
        )
    else:
        user_ids = np.repeat(np.array(user_df["user_id"]), np.array(user_df["freq"]))
        camp_ids = np.repeat(np.array(camp_df["camp_id"]), np.array(camp_df["freq"]))
        np.random.shuffle(user_ids)
        np.random.shuffle(camp_ids)
        obs_df = pd.DataFrame({"user_id": user_ids, "camp_id": camp_ids})
    obs_df = obs_df.merge(user_df.drop(columns="freq"), how="left", on="user_id")
    obs_df = obs_df.merge(camp_df.drop(columns="freq"), how="left", on="camp_id")
    if cross_weight is not None:
        cross_prod = np.einsum(
            "ij,ik->ijk",
            obs_df[["user_f0", "user_f1", "user_fh"]].values,
            obs_df[["camp_f0", "camp_f1", "camp_fh"]].values,
        )
        iprod = (cross_weight[np.newaxis, :, :] * cross_prod).sum(axis=(1, 2))
    else:
        iprod = (
            obs_df[["user_f0", "user_f1", "user_fh"]].values
            * obs_df[["camp_f0", "camp_f1", "camp_fh"]].values
        ).sum(axis=1)
    if response_sig_a is None:
        obs_df["response"] = iprod > 0
    else:
        obs_df["response"] = np.random.binomial(n=1, p=sigmoid(iprod, a=response_sig_a))
    return (
        obs_df.astype(
            {"cohort": "category", "user_id": "category", "camp_id": "category"}
        ),
        user_df.astype({"cohort": "category", "user_id": "category"}),
        camp_df.astype({"camp_id": "category"}),
    )
