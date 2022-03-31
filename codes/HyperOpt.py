import pandas as pd
import numpy as np
import os.path
from ai4water import Model
from ai4water.postprocessing.SeqMetrics import RegressionMetrics
from ai4water.hyperopt import Real, Integer, Categorical, HyperOpt
from ai4water.utils.utils import dateandtime_now
from ai4water.utils.utils import jsonize

# ai4water v. 1.0b5
print("Save folder / name")
PREFIX = f"conVlstm_Opt_runModel_{dateandtime_now()}"

print("Data load")
df = pd.read_excel('E:\Coding\Data\For test\cc_all.xlsx')
data = df

# setup inputs and outputs
inputs = ['Conductivity_in', 'pH_out', 'pH_in', 'Current', 'Voltage']
outputs = ['Conductivity_out']

print("objective_function")
def objective_func(**suggestions) -> float:

    suggestions = jsonize(suggestions)

    # suggested hyO
    filters     = suggestions['filters']   
    kernel_size = suggestions['kernel_size']  
    lstm0       = suggestions['lstm0']
    lstm1       = suggestions['lstm1']
    lookback    = suggestions['lookback']    
    batch_size  = suggestions['batch_size']     
    lr          = suggestions['lr']
    epochs      = suggestions['epochs']
    mp_size     = suggestions['mp_size']
    act_lstm0   = suggestions['act_lstm0']
    act_lstm1   = suggestions['act_lstm1']

    sub_sequences = 3 
    time_steps = lookback // sub_sequences

    # layer construction("CNN_LSTM model" as an example here)
    layers = {
    "Input":  {'shape': (lookback, len(inputs))},
    "Reshape":  {'target_shape': (sub_sequences, time_steps, len(inputs))},
    "TimeDistributed_0": {},
    "Conv1D_0": {'filters': int(filters),
     'kernel_size': int(kernel_size), 'name': 'first_conv1d', "padding": "same"},
    'LeakyReLU': {},
    # "TimeDistributed_1": {},
    # "Conv1D_1": {'config':  {'filters': 32, 'kernel_size': 2}},
    # 'ELU_1': {'config':  {}},
    # "TimeDistributed_2": {},
    # "Conv1D_2": {'config':  {'filters': 16, 'kernel_size': 2}},
    # 'tanh_2': {},
    "TimeDistributed_3": {},
    "MaxPool1D": {'pool_size': int(mp_size), "padding": "same"},
    "TimeDistributed_4": {},
    'Flatten': {},
    'LSTM_0': {'units': int(lstm0), 'activation': act_lstm0, 'dropout': 0.4, 'recurrent_dropout': 0.5,
                          'return_sequences': True,
               'name': 'LSTM_0'},
    'ReLU_1': {},
    'LSTM_1': {'units': int(lstm1), 'activation': act_lstm1, 'dropout': 0.4, 'recurrent_dropout': 0.5,
                          'name': 'LSTM_1'},
    'sigmoid_2': {},
    'Dense': 1
    }

    model = Model(
        model = {'layers': layers},
        input_features=inputs,   # columns in csv file to be used as input
        output_features=outputs,     # columns in csv file to be used as output
        split_random=True,
        seed=313,
        train_fraction=0.7,
        val_fraction=0.2,
        x_transformation = [{'method': 'log', 'features': ['Conductivity_in']}
                ],
        y_transformation = [{'method': 'log', 'features': ['Conductivity_out']}
                ],
        # allow_input_nans=True,
        ts_args={"lookback": lookback},
        batch_size=int(batch_size),
        lr=float(lr),
        epochs=int(epochs),
        # shuffle=True,
        prefix=PREFIX,
        verbosity=1
        )

    # set the name of training data
    x_train, y_train = model.training_data(data=df)
    print(f"x_train shape: {x_train.shape}")

    # Model Training
    history = model.fit(
        x=x_train,
        y=y_train,
        validation_split=0.2,
    )

    # Evaluate model
    # HyO toward low val
    val_score = np.min(history.history['val_loss'])

    return val_score

print("parameter space")
# Categorical(float+int), Real(float in range, 3.14), Integer(int in rage, 3)
param_space = [
    Categorical([32, 64, 128, 256], name="filters"),
    Categorical([2, 3, 4, 5], name="kernel_size"),
    Categorical([2, 3, 4], name="mp_size"),
    Categorical([8, 16, 32, 64, 128], name="lstm0"),
    Categorical([8, 16, 32, 64, 128], name="lstm1"),
    Categorical([6, 9, 12], name="lookback"),
    Categorical([8, 16, 32, 64, 128], name="batch_size"),
    Categorical(['relu', 'leakyrelu', 'elu', 'tanh'], name="act_lstm0"),
    Categorical(['relu', 'leakyrelu', 'elu', 'tanh'], name="act_lstm1"),
    Real(1e-5, 0.05, name="lr"),  
    Integer(50, 300, name="epochs") 
]

# Hyper-parameter Optimization option set
# set "num_iteration" / minimum =10, >25 (30~35) recommanded
optimizer = HyperOpt(
    algorithm="bayes",
    objective_fn=objective_func,
    param_space=param_space,
    num_iterations=35,
    opt_path=os.path.join(os.getcwd(), "results", PREFIX)
)

print("optimizer fit")
optimizer.fit()

# Print the optimized hyperparameters
print(optimizer.best_paras())