import itertools as itt

import numpy as np
import pandas as pd
import tensorflow as tf

# Makes panda and numpy easier to read
pd.set_option("display.float_format", lambda x: "%.3f" % x)
np.set_printoptions(precision=3, suppress=True)


def sigmoid(x, a=1, b=0):
    return 1 / (1 + np.exp(-a * (x + b)))


def unit_norm(x):
    return x / np.sqrt(np.sum(x**2, axis=1))[:, np.newaxis]


def generate_CSV(df, train_path, val_path, test_path, verbose=False, save_space=True):
    """
    This function accepts input file path & output paths to export
    training, validation and test CSVs.
    """
    # Set train & val set splits
    train_split_index = int(0.6 * len(df))
    val_split_index = int(0.8 * len(df))
    df = df.sample(frac=1, random_state=42).reset_index()
    train_df = df.loc[:train_split_index, :]
    val_df = df.loc[train_split_index:val_split_index, :]
    test_df = df.loc[val_split_index:, :]
    if verbose:
        # Printing the dataframes
        print(
            f"[INFO] The training dataframe has {len(train_df)} instances from indices {0} to {train_split_index - 1}"
        )
        train_df.head()
        print(
            f"[INFO] The validation dataframe has {len(val_df)} instances from indices {train_split_index} to {val_split_index - 1}"
        )
        val_df.head()
        print(
            f"[INFO] The training dataframe has {len(test_df)} instances from indices {val_split_index} to {len(df)}"
        )
        test_df.head()
    # Export training and test set into separate CSV files
    train_df.to_csv(train_path, index=False, header=True)
    val_df.to_csv(val_path, index=False, header=True)
    test_df.to_csv(test_path, index=False, header=True)
    if save_space:
        del df
        del train_df
        del val_df
        del test_df
    return


def df_to_dataloader(
    file_path, feature_columns, target, batch_size=32, shuffle=True, subtract=False
):
    """
    Given a df pattern, convert them to DL
    """
    use_cols = feature_columns + [target]
    dataframe = pd.read_csv(file_path, usecols=use_cols)[use_cols]
    labels = dataframe.pop(target)
    if subtract:
        labels -= 1
    labels = tf.one_hot(labels, depth=10)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    dl = ds.batch(batch_size)
    return dl


def generate_feature_columns(hidden_include=False):
    """
    Returns TF numeric, embedding & crossed feature columns & column inputs.
    """
    # Master return value dictionaries
    feature_column_dict = {}
    feature_column_input_dict = {}
    # Numeric Columns
    numeric_feature_columns = []
    numeric_feature_column_inputs = {}
    # If hidden_include variable is set to True ???> include the hidden categorical variables as well
    if not hidden_include:
        numeric_df_col_names = ["user_f0", "user_f1", "camp_f0", "camp_f1"]
    else:
        numeric_df_col_names = [
            "user_f0",
            "user_f1",
            "user_fh",
            "camp_f0",
            "camp_f1",
            "camp_fh",
        ]
    for col_name in numeric_df_col_names:
        tf_nfc = tf.feature_column.numeric_column(col_name)
        numeric_feature_columns.append(tf_nfc)
        numeric_feature_column_inputs[col_name] = tf.keras.Input(
            shape=(), name=col_name
        )
    feature_column_dict["numeric"] = numeric_feature_columns
    feature_column_input_dict["numeric"] = numeric_feature_column_inputs
    # Embedding Columns
    embedding_feature_columns = []
    embedding_feature_column_inputs = {}
    user_id = tf.feature_column.categorical_column_with_hash_bucket(
        "user_id", hash_bucket_size=1000, dtype=tf.int64
    )
    campaign_id = tf.feature_column.categorical_column_with_hash_bucket(
        "camp_id", hash_bucket_size=100, dtype=tf.int64
    )
    cohort = tf.feature_column.categorical_column_with_hash_bucket(
        "cohort", hash_bucket_size=10, dtype=tf.int64
    )
    user_id_embedding = tf.feature_column.embedding_column(user_id, dimension=16)
    embedding_feature_columns.append(user_id_embedding)
    embedding_feature_column_inputs["user_id"] = tf.keras.Input(
        shape=(), name="user_id", dtype=tf.int64
    )
    campaign_id_embedding = tf.feature_column.embedding_column(campaign_id, dimension=7)
    embedding_feature_columns.append(campaign_id_embedding)
    embedding_feature_column_inputs["camp_id"] = tf.keras.Input(
        shape=(), name="camp_id", dtype=tf.int64
    )
    cohort_embedding = tf.feature_column.embedding_column(cohort, dimension=7)
    embedding_feature_columns.append(cohort_embedding)
    embedding_feature_column_inputs["cohort"] = tf.keras.Input(
        shape=(), name="cohort", dtype=tf.int64
    )
    feature_column_dict["embedding"] = embedding_feature_columns
    feature_column_input_dict["embedding"] = embedding_feature_column_inputs
    # Crossed Columns
    user_campaign_cross = tf.feature_column.crossed_column(
        ["user_id", "camp_id"], hash_bucket_size=100000
    )
    user_campaign_cross_col = [tf.feature_column.indicator_column(user_campaign_cross)]
    feature_column_dict["crossed"] = user_campaign_cross_col
    return feature_column_dict, feature_column_input_dict


def norm(x):
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    return (x - xmin) / (xmax - xmin)


def enumerated_product(*args):
    yield from zip(itt.product(*(range(len(x)) for x in args)), itt.product(*args))
