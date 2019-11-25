import os
import sys
import pickle
from copy import deepcopy
from collections import defaultdict
from pprint import pprint
import numpy as np
import pandas as pd

from sklearn.metrics import log_loss, mean_squared_error


def import_data(data):    
    return pd.read_csv("train_test_data/" + data + ".csv", index_col="uid").drop("Unnamed: 0", axis="columns")


X_train = import_data("X_train")
X_test = import_data("X_test")
y_train = import_data("y_train")
y_test = import_data("y_test")

y_train = y_train[[e for e in y_train.columns if 'post' in e]]
y_test  =  y_test[[e for e in y_test.columns  if 'post' in e]]
train_cols = X_train.columns.tolist()
target_cols = y_train.columns.tolist()

y_test.shape

# remove rows where targets all all NA
train = pd.concat([y_train, X_train], axis = 1)
train = train.dropna(how = 'all', subset = ['panas_pos_imp_post', 'panas_neg_imp_post', 'panas_pos_imp_post'])
test  = pd.concat([y_test, X_test], axis = 1)
test = test.dropna(how = 'all', subset = ['panas_pos_imp_post', 'panas_neg_imp_post', 'panas_pos_imp_post'])

test_score = dict()
for target in target_cols:
    if '_class_' in target:
        test_score[target] = log_loss(y_test[target], np.repeat(y_train[target].mean(), y_test.shape[0]))
    else:
        test_score[target] = mean_squared_error(y_test[target], np.repeat(y_train[target].mean(), y_test.shape[0]))

for target in target_cols:
    print(f'target {target} {"Log loss" if "_class_" in target else "MSE"} = {test_score[target]:.4f}')