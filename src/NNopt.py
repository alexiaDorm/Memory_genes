import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import pyreadr

from load_data import open_charac, add_general_charac
from binaryclass_memory import *
    

torch.manual_seed(1)

fused = load_charac()
objective = Objective(fused)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

best_trial = study.best_trial

for key, value in best_trial.params.items():
    print("{}: {}".format(key, value))