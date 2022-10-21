from copy import copy

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
from sklearn.decomposition import PCA

def build_wide_model(
    feature_column_dict,
    inputs,
    wmodel_dir,
    inc_numeric=False,
    name="wmodel.h5",
    ckpt_name="wmodel_checkpoint.h5",
):
    """
    Builds and returns a wide-only model
    """
    # Should we send numeric columns as well?
    if inc_numeric:
        wide_only_feature_columns = (
            feature_column_dict["numeric"] + feature_column_dict["crossed"]
        )
    else:
        wide_only_feature_columns = feature_column_dict["crossed"]

    wmodel_path = wmodel_dir + "/" + name
    wmodel_checkpoint_path = wmodel_dir + "/" + ckpt_name
    # Simple early stopping
    w_es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", verbose=1, patience=5
    )
    w_mc = tf.keras.callbacks.ModelCheckpoint(
        str(wmodel_checkpoint_path),
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
    )
    # Build the Wide Only Model
    w_feature_layer = tf.keras.layers.DenseFeatures(
        wide_only_feature_columns, name="dense_feature_layer"
    )(inputs)
    w_linear_layer = tf.keras.layers.Dense(
        10, activation="softmax", name="wide_output"
    )(w_feature_layer)
    wmodel = tf.keras.Model(inputs=inputs, outputs=w_linear_layer)
    wmodel.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC()],
    )
    return wmodel, wmodel_path, w_es, w_mc


def build_deep_model(
    feature_column_dict,
    inputs,
    dmodel_dir,
    name="dmodel.h5",
    ckpt_name="dmodel_checkpoint.h5",
):

    """
    Builds and returns a deep-only model
    """
    # Passed object is a list instead of a dictionary
    if isinstance(feature_column_dict, list):
        deep_only_feature_columns = feature_column_dict[:]
    else:
        deep_only_feature_columns = []
        for feature_list in feature_column_dict.values():
            deep_only_feature_columns.extend(feature_list)
    deep_only_hidden_units = [512, 256, 128]
    dmodel_path = dmodel_dir + "/" + name
    dmodel_checkpoint_path = dmodel_dir + "/" + ckpt_name
    # simple early stopping
    d_es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", verbose=1, patience=20
    )
    d_mc = tf.keras.callbacks.ModelCheckpoint(
        str(dmodel_checkpoint_path),
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
    )
    # Build the Wide Only Model
    d_feature_layer = tf.keras.layers.DenseFeatures(
        deep_only_feature_columns, name="dense_feature_layer"
    )(inputs)
    d_hidden_layer_1 = tf.keras.layers.Dense(
        deep_only_hidden_units[0], activation="relu", name="fc_1"
    )(d_feature_layer)
    d_hidden_layer_2 = tf.keras.layers.Dense(
        deep_only_hidden_units[1], activation="relu", name="fc_2"
    )(d_hidden_layer_1)
    d_hidden_layer_3 = tf.keras.layers.Dense(
        deep_only_hidden_units[2], activation="relu", name="fc_3"
    )(d_hidden_layer_2)
    d_output_layer = tf.keras.layers.Dense(
        10, activation="softmax", name="deep_output"
    )(d_hidden_layer_3)
    dmodel = tf.keras.Model(inputs=inputs, outputs=d_output_layer)
    dmodel.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC()],
    )
    return dmodel, dmodel_path, d_es, d_mc


def build_wide_and_deep_model(
    feature_column_dict,
    inputs,
    wdmodel_dir,
    name="wdmodel.h5",
    ckpt_name="wdmodel_checkpoint.h5",
):

    """
    Builds and returns a wide & deep model
    """
    wide_wd_feature_columns = feature_column_dict["crossed"]
    deep_wd_feature_columns = (
        feature_column_dict["numeric"] + feature_column_dict["embedding"]
    )
    deep_wd_hidden_units = [512, 256, 128]
    wdmodel_path = wdmodel_dir + "/" + name
    wdmodel_checkpoint_path = wdmodel_dir + "/" + ckpt_name
    # simple early stopping
    wd_es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", verbose=1, patience=20
    )
    wd_mc = tf.keras.callbacks.ModelCheckpoint(
        str(wdmodel_checkpoint_path),
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
    )
    # Build the Wide Model
    w_feature_layer = tf.keras.layers.DenseFeatures(
        wide_wd_feature_columns, name="wide_feature_layer"
    )(inputs)
    # Build the Deep Model
    d_feature_layer = tf.keras.layers.DenseFeatures(
        deep_wd_feature_columns, name="deep_feature_layer"
    )(inputs)
    d_hidden_layer_1 = tf.keras.layers.Dense(
        deep_wd_hidden_units[0], activation="relu", name="deep_fc_1"
    )(d_feature_layer)
    d_hidden_layer_2 = tf.keras.layers.Dense(
        deep_wd_hidden_units[1], activation="relu", name="deep_fc_2"
    )(d_hidden_layer_1)
    d_hidden_layer_3 = tf.keras.layers.Dense(
        deep_wd_hidden_units[2], activation="relu", name="deep_fc_3"
    )(d_hidden_layer_2)
    # Combine the Wide & Deep
    wd_both = tf.keras.layers.concatenate([w_feature_layer, d_hidden_layer_3])
    wd_output_layer = tf.keras.layers.Dense(
        10, activation="softmax", name="deep_output"
    )(wd_both)
    wd_model = tf.keras.Model(inputs=inputs, outputs=wd_output_layer)
    wd_model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC()],
    )
    return wd_model, wdmodel_path, wd_es, wd_mc


def build_bayesian_model(
    feature_column_dict,
    inputs,
    bayesian_dir,
    p=0.3,
    name="bmodel.h5",
    ckpt_name="bmodel_checkpoint.h5",
):

    """
    Builds and returns a wide & deep bayesian bandit model via MC dropout
    """
    bayesian_path = bayesian_dir + "/" + name
    bayesian_checkpoint_path = bayesian_dir + "/" + ckpt_name
    bayesian_wide_feature_columns = feature_column_dict["crossed"]
    bayesian_deep_feature_columns = (
        feature_column_dict["numeric"] + feature_column_dict["embedding"]
    )
    bayesian_deep_hidden_units = [512, 256, 128]
    # simple early stopping
    bayesian_es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", verbose=1, patience=10
    )
    bayesian_mc = tf.keras.callbacks.ModelCheckpoint(
        str(bayesian_checkpoint_path),
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
    )
    # Build the Wide Model
    w_feature_layer = tf.keras.layers.DenseFeatures(
        bayesian_wide_feature_columns, name="wide_feature_layer"
    )(inputs)
    # Build the Deep Model
    d_feature_layer = tf.keras.layers.DenseFeatures(
        bayesian_deep_feature_columns, name="deep_feature_layer"
    )(inputs)
    d_hidden_layer_1 = tf.keras.layers.Dense(
        bayesian_deep_hidden_units[0], activation="relu", name="deep_fc_1"
    )(d_feature_layer)
    d_hidden_layer_2 = tf.keras.layers.Dense(
        bayesian_deep_hidden_units[1], activation="relu", name="deep_fc_2"
    )(d_hidden_layer_1)
    d_hidden_layer_3 = tf.keras.layers.Dense(
        bayesian_deep_hidden_units[2], activation="relu", name="deep_fc_3"
    )(d_hidden_layer_2)
    # Combine the Wide & Deep
    bayesian_both = tf.keras.layers.concatenate(
        [w_feature_layer, d_hidden_layer_3], name="concatenate"
    )  # Name is concatenate_1 for this
    # bayesian_pre_multihead_dropout = tf.keras.layers.Dropout(0.3)(bayesian_both, training=True) - Commented out to solve OOM issues
    bayesian_multihead = tf.keras.layers.Dense(64, activation="relu", name="multihead")(
        bayesian_both
    )
    bayesian_dropout = tf.keras.layers.Dropout(0.3, name="dropout")(
        bayesian_multihead, training=True
    )
    bayesian_output_layer = tf.keras.layers.Dense(
        10, activation="softmax", name="bayesian_model_output"
    )(bayesian_dropout)
    bayesian_model = tf.keras.Model(inputs=inputs, outputs=bayesian_output_layer)
    bayesian_model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC()],
    )
    return bayesian_model, bayesian_path, bayesian_es, bayesian_mc


def find_ucb(raw_samples_l, value_index):
    raw_samples_array = np.array(raw_samples_l)
    # Array is 3D - 0: Samples, 1: Row, 2: Column
    # Sort the array along 0th axis
    raw_samples_array[::-1].sort(axis=0)
    # Pick the 5th largest value
    ucb_batch = raw_samples_array[value_index - 1]
    # Return the value
    return ucb_batch


def evaluate_bandit(test_model, dl, num_of_samples=100):
    """
    Returns the TS and UCB accuracy for model on dl
    """
    # Constructing a intermediate wide and deep model to get the static concatenated outputs
    layer_name = "multihead"
    till_multihead_model = tf.keras.Model(
        inputs=test_model.input, outputs=test_model.get_layer(layer_name).output
    )
    # Now, constructing the multi-head that the concatenated input has to run through
    dropout_input = tf.keras.Input(
        shape=(64,), dtype=tf.float32, name="multihead_output"
    )
    next_layer = dropout_input
    for layer in test_model.layers[13:]:
        if "dropout" in layer.name:
            next_layer = layer(next_layer, training=True)
        else:
            next_layer = layer(next_layer)
    post_multihead_model = tf.keras.Model(inputs=dropout_input, outputs=next_layer)
    # Setting up variables to store running outputs
    value_index = num_of_samples - int(0.95 * num_of_samples)
    batch_cnt = 0
    # Variable to store comparison between bandit outputs & labels
    ts_bandit_output_l = []
    ucb_bandit_output_l = []
    # Iterate through the Batched Dataset, one batch at a time
    for features, labels in dl:
        if (batch_cnt + 1) % 5 == 0:
            print(f"[INFO] Working on Batch #{batch_cnt + 1}")
        # Generate predictions of the model for a given batch
        multihead_output = till_multihead_model.predict(features)
        ucb_working_l = []
        for multi_cnt in range(num_of_samples):
            if not ((batch_cnt + 1) % 5) and not ((multi_cnt + 1) % 5):
                print(f"[INFO] Drawing sample #{multi_cnt + 1} from the Posterior...")
            output = post_multihead_model.predict(multihead_output)
            ucb_working_l.append(output)  # Now begin drawing multiple samples for UCB
            if not (multi_cnt):  # Draw only one sample for TS
                ts_batch = output
                # ts_bandit_output_l.append(output)
        ucb_batch = find_ucb(ucb_working_l, value_index)
        # ucb_bandit_output_l.append(ucb_batch)
        ts_batch_class = np.argmax(ts_batch, axis=1)
        ucb_batch_class = np.argmax(ucb_batch, axis=1)
        # Compare output batch of class labels to label_batch
        ts_bandit_output_l.append(ts_batch_class == labels.numpy())
        ucb_bandit_output_l.append(ucb_batch_class == labels.numpy())
        batch_cnt += 1
    ts_bandit_output = np.concatenate(ts_bandit_output_l, axis=0)
    del ts_bandit_output_l
    ucb_bandit_output = np.concatenate(ucb_bandit_output_l, axis=0)
    del ucb_bandit_output_l
    return np.mean(ts_bandit_output), np.mean(ucb_bandit_output)


def cluster_xgb(
    df,
    feats,
    resp="response",
    n_cohort=None,
    run_cluster=False,
    cluster_feats=["user_f0", "user_f1"],
    cluster_model=None,
    xgb_model=None,
    xgb_warm_start=False,
    **kwargs,
):
    if run_cluster:
        if cluster_model is None:
            cluster_model = KMeans(n_clusters=n_cohort)
            cluster_model.fit(df[cluster_feats])
        df["cohort"] = cluster_model.predict(df[cluster_feats])
    X = pd.get_dummies(df[feats])
    y = df[resp]
    if xgb_model is None:
        xgb_model = XGBClassifier(**kwargs)
        xgb_model.fit(X, y)
    elif xgb_warm_start:
        xgb_model.fit(X, y, xgb_model=xgb_model.get_booster())
    score = xgb_model.score(X, y)
    return score, cluster_model, xgb_model


class CohortXGB:
    def __init__(
        self,
        n_cohort,
        feats,
        resp="response",
        cohort_feats=None,
        n_neighbors=1,
        use_cohort_resp=False,
        **kwargs,
    ) -> None:
        self.cluster_model = KMeans(n_clusters=n_cohort)
        self.cohort_model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.xgb_model = XGBClassifier(**kwargs)
        self.use_cohort_resp = use_cohort_resp
        self.feats = feats
        self.resp = resp
        self.cohort_feats = cohort_feats

    def fit(self, df):
        if self.cohort_feats is not None:
            cluster_feats = copy(self.cohort_feats)
            cohort_df = df[self.cohort_feats + ["user_id"]].drop_duplicates()
            if self.use_cohort_resp:
                camps = np.sort(df["camp_id"].unique()).tolist()
                df_agg = (
                    df.groupby(["user_id", "camp_id"], observed=True)["response"]
                    .mean()
                    .reset_index()
                )
                df_agg["response"] = np.around(df_agg["response"]).astype(int)
                resp_df = df_agg.pivot(
                    index="user_id", columns="camp_id", values="response"
                ).reset_index()
                imputer = KNNImputer(n_neighbors=3)
                resp_df[camps] = imputer.fit_transform(resp_df[camps])
                cohort_df = cohort_df.merge(resp_df, on="user_id", how="left")
                cluster_feats += camps
            cohort_df["cohort"] = self.cluster_model.fit_predict(
                cohort_df[cluster_feats]
            )
            self.cohort_model.fit(cohort_df[self.cohort_feats], cohort_df["cohort"])
            df["cohort"] = pd.Categorical(
                cohort_df.set_index("user_id").loc[df["user_id"]]["cohort"],
                categories=df["cohort"].values.categories,
            )
        self.xgb_model.fit(pd.get_dummies(df[self.feats]), df[self.resp])

    def predict(self, df):
        if self.cohort_feats is not None:
            df["cohort"] = pd.Categorical(
                self.cohort_model.predict(df[self.cohort_feats]),
                categories=df["cohort"].values.categories,
            )
        return self.xgb_model.predict(pd.get_dummies(df[self.feats]))

    def score(self, df):
        y_prd = self.predict(df)
        return accuracy_score(df[self.resp], y_prd)


def AddPCA(df, n_campaigns):
    df = df.sort_values(["camp_id", "user_id"])
    resp_ls = np.column_stack([resp["response"] for resp in np.array_split(df, n_campaigns)])
    pca_resp = PCA(n_components=3).fit_transform(resp_ls)

    user_ids = np.array(df["user_id"].unique())
    pca_df = pd.DataFrame(
        {"user_id": user_ids, "PCA_1": pca_resp[:, 0], "PCA_2": pca_resp[:, 1], "PCA_3": pca_resp[:, 2]})

    df = df.merge(pca_df, how="left", on="user_id")

    return df, pca_df


def response_cluster(df, cohort_feats, n_cohort, n_neighbors=1):
    cluster_model = KMeans(n_clusters=n_cohort)
    cohort_model = KNeighborsClassifier(n_neighbors=n_neighbors)

    cluster_feats = copy(cohort_feats)
    cohort_df = df[cohort_feats + ["user_id"]].drop_duplicates()

    camps = np.sort(df["camp_id"].unique()).tolist()
    df_agg = (
        df.groupby(["user_id", "camp_id"], observed=True)["response"]
        .mean()
        .reset_index()
    )
    df_agg["response"] = np.around(df_agg["response"]).astype(int)
    resp_df = df_agg.pivot(
        index="user_id", columns="camp_id", values="response"
    ).reset_index()
    imputer = KNNImputer(n_neighbors=3)
    resp_df[camps] = imputer.fit_transform(resp_df[camps])
    cohort_df = cohort_df.merge(resp_df, on="user_id", how="left")
    cluster_feats += camps

    cohort_df["cohort"] = cluster_model.fit_predict(
        cohort_df[cluster_feats].values
    )
    cohort_model.fit(cohort_df[cohort_feats].values, cohort_df["cohort"].values)
    df["cohort"] = pd.Categorical(
        cohort_df.set_index("user_id").loc[df["user_id"]]["cohort"],
        categories=df["cohort"].values.categories,
    )

    df = df[["user_id", "cohort"]].drop_duplicates(subset=["user_id"]).rename(
        columns={"cohort": "cohort_response_cluster"})

    return df
