import gc
import importlib
import os
from routine.utilities import generate_CSV, df_to_dataloader, generate_feature_columns
from routine.models import build_wide_model, build_deep_model, build_wide_and_deep_model, \
    build_bayesian_model, evaluate_bandit
from os.path import exists
from pprint import pprint
import tensorflow as tf
import sys

DATA = './modelinputs/gen_data/observation.csv'
INPUT_DATA_PATH = './modelinputs/input_data'

if not exists(DATA):
    raise RuntimeError('observation.csv file not yet created...')
else:
    if not os.path.isdir(INPUT_DATA_PATH):
        os.makedirs(INPUT_DATA_PATH)

#%% Creating the training, validation and testing data for the model
train_path = INPUT_DATA_PATH + "/train.csv"
val_path = INPUT_DATA_PATH + "/val.csv"
test_path = INPUT_DATA_PATH + "/test.csv"
re_create = True
if re_create:
    generate_CSV(DATA,
                 train_path,
                 val_path,
                 test_path,
                 verbose=True)

#%% Preparing dataset for evaluation
batch_size = 100
n_epochs = 100
feature_columns = ["user_id", "camp_id", "cohort",
                   "user_f0", "user_f1", "user_fh",
                   "camp_f0", "camp_f1", "camp_fh"]
target_column = "response"

train_dl = df_to_dataloader(train_path,
                            feature_columns,
                            target_column,
                            batch_size=batch_size)
val_dl = df_to_dataloader(val_path,
                          feature_columns,
                          target_column,
                          batch_size=batch_size)
test_dl = df_to_dataloader(test_path,
                           feature_columns,
                           target_column,
                           shuffle=False,
                           batch_size=batch_size)

print("[INFO] Train dataloader:")
pprint(train_dl)
print("[INFO] Val dataloader:")
pprint(val_dl)
print("[INFO] Test dataloader:")
pprint(test_dl)


#%% Creating TF feature columns
feature_column_dict, feature_column_input_dict = generate_feature_columns()
# defining the input to be fed into each model
inputs = {**feature_column_input_dict["numeric"], **feature_column_input_dict["embedding"]}


#%% Models
models_dir = './model_checkpoint'
if not os.path.isdir(models_dir):
    os.makedirs(models_dir)
# create the folders to save the checkpoints
wmodel_dir = models_dir + '/Wide'
dmodel_dir = models_dir + '/Deep'
wdmodel_dir = models_dir + '/W&D'
bayesian_dir = models_dir + '/Bayesian'
os.makedirs(wmodel_dir, exist_ok=True)
os.makedirs(dmodel_dir, exist_ok=True)
os.makedirs(wdmodel_dir, exist_ok=True)
os.makedirs(bayesian_dir, exist_ok=True)
# setting the hyperparameters
lr = 1e-3
gc.collect()

#%% Wide only model
wmodel, wmodel_path, w_es, w_mc = build_wide_model(feature_column_dict,
                                                   inputs,
                                                   wmodel_dir=wmodel_dir)
wmodel.summary()  # To display the architecture

#%% Training already done - Just load the model!
again_training = True
if again_training:
    # create callback for model saving
    w_m = tf.keras.callbacks.ModelCheckpoint(
        filepath=wmodel_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min')
    H = wmodel.fit(train_dl,
                   batch_size=batch_size,
                   epochs=n_epochs,
                   validation_data=val_dl,
                   shuffle=False,
                   validation_batch_size=batch_size,
                   callbacks=[w_es, w_mc, w_m])
else:
    wmodel = tf.keras.models.load_model(wmodel_path)

#%% Generate the predictions
eval_wmodel_train = wmodel.evaluate(train_dl)
eval_wmodel_val = wmodel.evaluate(val_dl)
eval_wmodel_test = wmodel.evaluate(test_dl)
# Print the results
print("\n[INFO] On Training Set:")
print(eval_wmodel_train)
print("\n[INFO] On Validation Set:")
print(eval_wmodel_val)
print("\n[INFO] On Test Set:")
print(eval_wmodel_test)

# Deep only model
#%% With only embeddings
dmodel_1_emb, dmodel_1_emb_path, d1_es, d1_mc = build_deep_model(feature_column_dict["embedding"],
                                                                 inputs,
                                                                 dmodel_dir,
                                                                 name="dmodel_1_emb.h5",
                                                                 ckpt_name="dmodel_1_emb_checkpoint.h5")
dmodel_1_emb.summary()  # To display the architecture

#%% Training already done - Just load the model!
again_training = True
if again_training:
    # create callback for model saving
    d1_m = tf.keras.callbacks.ModelCheckpoint(
        filepath=dmodel_1_emb_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min')
    H1 = dmodel_1_emb.fit(train_dl,
                          batch_size=batch_size,
                          epochs=n_epochs,
                          validation_data=val_dl,
                          shuffle=False,
                          validation_batch_size=batch_size,
                          callbacks=[d1_es, d1_mc, d1_m])
else:
    dmodel_1_emb = tf.keras.models.load_model(dmodel_1_emb_path)

#%% Generate predictions on train, val & test set
eval_dmodel_1_emb_train = dmodel_1_emb.evaluate(train_dl, batch_size=batch_size)
eval_dmodel_1_emb_val = dmodel_1_emb.evaluate(val_dl, batch_size=batch_size)
eval_dmodel_1_emb_test = dmodel_1_emb.evaluate(test_dl, batch_size=batch_size)
# Print the results
print("\n[INFO] On Training Set:")
print(eval_dmodel_1_emb_train)
print("\n[INFO] On Validation Set:")
print(eval_dmodel_1_emb_val)
print("\n[INFO] On Test Set:")
print(eval_dmodel_1_emb_test)



#%% With only numeric features
dmodel_2_num, dmodel_2_num_path, d2_es, d2_mc = build_deep_model(feature_column_dict["numeric"],
                                                                 inputs,
                                                                 dmodel_dir,
                                                                 name="dmodel_2_num.h5",
                                                                 ckpt_name="dmodel_2_num_checkpoint.h5")
dmodel_2_num.summary()

#%% Training already done - Just load the model!
again_training = True
if again_training:
    # create callback for model saving
    d2_m = tf.keras.callbacks.ModelCheckpoint(
        filepath=dmodel_2_num_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min')
    H2 = dmodel_2_num.fit(train_dl,
                          batch_size=batch_size,
                          epochs=n_epochs,
                          validation_data=val_dl,
                          shuffle=False,
                          validation_batch_size=batch_size,
                          callbacks=[d2_es, d2_mc, d2_m])
else:
    dmodel_2_num = tf.keras.models.load_model(dmodel_2_num_path)

#%% Generate predictions on train, val & test set
eval_dmodel_2_num_train = dmodel_2_num.evaluate(train_dl, batch_size=batch_size)
eval_dmodel_2_num_val = dmodel_2_num.evaluate(val_dl, batch_size=batch_size)
eval_dmodel_2_num_test = dmodel_2_num.evaluate(test_dl, batch_size=batch_size)
# Print the results
print("\n[INFO] On Training Set:")
print(eval_dmodel_2_num_train)
print("\n[INFO] On Validation Set:")
print(eval_dmodel_2_num_val)
print("\n[INFO] On Test Set:")
print(eval_dmodel_2_num_test)



#%% With embeddings and numeric features
dmodel_3_num_emb, dmodel_3_num_emb_path, d3_es, d3_mc = build_deep_model(feature_column_dict,
                                                                         inputs,
                                                                         dmodel_dir,
                                                                         name="dmodel_3_num_emb.h5",
                                                                         ckpt_name="dmodel_3_num_emb_checkpoint.h5")
dmodel_3_num_emb.summary()

#%% Training already done - Just load the model!
again_training = True
if again_training:
    # create callback for model saving
    d3_m = tf.keras.callbacks.ModelCheckpoint(
        filepath=dmodel_3_num_emb_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min')
    H3 = dmodel_3_num_emb.fit(train_dl,
                              batch_size=batch_size,
                              epochs=n_epochs,
                              validation_data=val_dl,
                              shuffle=False,
                              validation_batch_size=batch_size,
                              callbacks=[d3_es, d3_mc, d3_m])
else:
    dmodel_3_num_emb = tf.keras.models.load_model(dmodel_3_num_emb_path)

#%% Generate predictions on train, val & test set
eval_dmodel_3_num_emb_train = dmodel_3_num_emb.evaluate(train_dl, batch_size=batch_size)
eval_dmodel_3_num_emb_val = dmodel_3_num_emb.evaluate(val_dl, batch_size=batch_size)
eval_dmodel_3_num_emb_test = dmodel_3_num_emb.evaluate(test_dl, batch_size=batch_size)
# Print the results
print("\n[INFO] On Training Set:")
print(eval_dmodel_3_num_emb_train)
print("\n[INFO] On Validation Set:")
print(eval_dmodel_3_num_emb_val)
print("\n[INFO] On Test Set:")
print(eval_dmodel_3_num_emb_test)



#%% With normal and hidden numeric features
# Get the new feature column and input dicts
feature_column_dict_hidden, feature_column_input_dict_hidden = generate_feature_columns(hidden_include=True)
inputs_hidden = {**feature_column_input_dict_hidden["numeric"], **feature_column_input_dict_hidden["embedding"]}
dmodel_4_hid, dmodel_4_hid_path, d4_es, d4_mc = build_deep_model(feature_column_dict_hidden,
                                                                 inputs_hidden,
                                                                 dmodel_dir,
                                                                 name="dmodel_4_hid.h5",
                                                                 ckpt_name="dmodel_4_hid_checkpoint.h5")
dmodel_4_hid.summary()

#%% Training already done - Just load the model!
again_training = True
if again_training:
    # create callback for model saving
    d4_m = tf.keras.callbacks.ModelCheckpoint(
        filepath=dmodel_4_hid_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min')
    H4 = dmodel_4_hid.fit(train_dl,
                          batch_size=batch_size,
                          epochs=n_epochs,
                          validation_data=val_dl,
                          shuffle=False,
                          validation_batch_size=batch_size,
                          callbacks=[d4_es, d4_mc, d4_m])
else:
    dmodel_4_hid = tf.keras.models.load_model(dmodel_4_hid_path)

#%% Generate predictions on train, val & test set
eval_dmodel_4_hid_train = dmodel_4_hid.evaluate(train_dl, batch_size=batch_size)
eval_dmodel_4_hid_val = dmodel_4_hid.evaluate(val_dl, batch_size=batch_size)
eval_dmodel_4_hid_test = dmodel_4_hid.evaluate(test_dl, batch_size=batch_size)
# Print the results
print("\n[INFO] On Training Set:")
print(eval_dmodel_4_hid_train)
print("\n[INFO] On Validation Set:")
print(eval_dmodel_4_hid_val)
print("\n[INFO] On Test Set:")
print(eval_dmodel_4_hid_test)

#%% Wide & deep model
wdmodel, wdmodel_path, wd_es, wd_mc = build_wide_and_deep_model(feature_column_dict,
                                                                inputs,
                                                                wdmodel_dir=wdmodel_dir)
wdmodel.summary()  # To display the architecture

#%% Training already done - Just load the model!
again_training = True
if again_training:
    # create callback for model saving
    wd_m = tf.keras.callbacks.ModelCheckpoint(
        filepath=wdmodel_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min')
    H = wdmodel.fit(train_dl,
                    batch_size=batch_size,
                    epochs=n_epochs,
                    validation_data=val_dl,
                    shuffle=False,
                    validation_batch_size=batch_size,
                    callbacks=[wd_es, wd_mc, wd_m])
else:
    wdmodel = tf.keras.models.load_model(wdmodel_path)

#%% Generate predictions on train, val & test set
eval_wdmodel_train = wdmodel.evaluate(train_dl, batch_size=batch_size)
eval_wdmodel_val = wdmodel.evaluate(val_dl, batch_size=batch_size)
eval_wdmodel_test = wdmodel.evaluate(test_dl, batch_size=batch_size)
# Print the results
print("\n[INFO] On Training Set:")
print(eval_wdmodel_train)
print("\n[INFO] On Validation Set:")
print(eval_wdmodel_val)
print("\n[INFO] On Test Set:")
print(eval_wdmodel_test)

#%% Bayesian Wide & deep model
bmodel, bmodel_path, b_es, b_mc = build_bayesian_model(feature_column_dict,
                                                       inputs,
                                                       bayesian_dir)
bmodel.summary()  # To display the architecture

#%% Training already done - Just load the model!
again_training = True
if again_training:
    # create callback for model saving
    b_m = tf.keras.callbacks.ModelCheckpoint(
        filepath=bmodel_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min')
    H = bmodel.fit(train_dl,
                   batch_size=batch_size,
                   epochs=n_epochs,
                   validation_data=val_dl,
                   shuffle=False,
                   validation_batch_size=batch_size,
                   callbacks=[b_es, b_mc, b_m])
else:
    bmodel = tf.keras.models.load_model(bmodel_path)

#%% Generate predictions on train, val & test set
ts_train, ucb_train = evaluate_bandit(bmodel, train_dl)
ts_val, ucb_val = evaluate_bandit(bmodel, val_dl)
ts_test, ucb_test = evaluate_bandit(bmodel, test_dl)
# Print the results
print("\nUCB\n[INFO] On Training Set:")
print(ucb_train)
print("\n[INFO] On Validation Set:")
print(ucb_val)
print("\n[INFO] On Test Set:")
print(ucb_test)

# Print the results
print("\nThompson Sampling\n[INFO] On Training Set:")
print(ts_train)
print("\n[INFO] On Validation Set:")
print(ts_val)
print("\n[INFO] On Test Set:")
print(ts_test)
