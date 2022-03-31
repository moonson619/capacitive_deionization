import pandas as pd
import numpy as np
import tensorflow as tf
# from tensorflow import keras
from ai4water import Model
# from ai4water.postprocessing.SeqMetrics import RegressionMetrics
# from ai4water.hyperopt import Real, Integer, Categorical, HyperOpt

# ai4water v. 1.0b5
# results will be automatically saved under "result" folder with date-time.

# ignore version error
tf.compat.v1.disable_eager_execution()

# load raw data
df = pd.read_excel('E:\Coding\Data\For test\cc_all.xlsx')
data = df

# load model config and optimized hyperparameter
model = Model.from_config_file(
    "E:\\Coding\\ai4water_V_1b5_code\\results\\conVlstm_Opt_runModel_20220321_123714_cc_all_USE\\1_20220318_144541\\config.json")

# load trained weights
model.update_weights(
    "E:\\Coding\\ai4water_V_1b5_code\\results\\conVlstm_Opt_runModel_20220321_123714_cc_all_USE\\1_20220318_144541\\weights\\weights_083_0.00075.hdf5")

# prediction
print('Predictions started')
test_true, test_pred = model.predict(data=df, return_true=True)
train_true, train_pred = model.predict(data='training', return_true=True)
val_true, val_pred = model.predict(data='validation', return_true=True)