import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import pyreadr

from load_data import open_charac, add_general_charac
from binaryclass_memory import *
    

torch.manual_seed(1)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=200)

best_trial = study.best_trial

for key, value in best_trial.params.items():
    print("{}: {}".format(key, value))