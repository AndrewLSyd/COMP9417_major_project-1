# -*- coding: utf-8 -*-
"""
COMP9417 group assignment
Purpose: merges all the engineered features and creates a train test split
of the data
Author: Andrew Lau
"""
import os
import pandas as pd
import numpy as np
from random import sample, seed
from math import ceil

# CONSTANTS -------------------------------------------------------------------
FPATH_PREPROCESSED = "preprocessed_data"
FPATH_PROCESSED = "processed_data"
TEST_SIZE = 0.2
RANDOM_STATE = 123
TARGET_LIST = ["panas_pos_raw_pre", "panas_neg_raw_pre", 
               "flourishing_scale_raw_pre", "panas_pos_imp_pre",
               "panas_neg_imp_pre", "flourishing_scale_imp_pre",
               "panas_pos_raw_class_pre", "panas_neg_raw_class_pre",
               "flourishing_scale_raw_class_pre", "panas_pos_imp_class_pre",
               "panas_neg_imp_class_pre", "flourishing_scale_imp_class_pre",
               "panas_pos_raw_post", "panas_neg_raw_post", 
               "flourishing_scale_raw_post", "panas_pos_imp_post", 
               "panas_neg_imp_post", "flourishing_scale_imp_post",
               "panas_pos_raw_class_post", "panas_neg_raw_class_post",
               "flourishing_scale_raw_class_post", "panas_pos_imp_class_post",
               "panas_neg_imp_class_post", "flourishing_scale_imp_class_post"]

# DATA IMPORT -----------------------------------------------------------------
# list all files in the features folder
df_list = [pd.read_csv(os.path.join(FPATH_PREPROCESSED, file)) for file in 
             os.listdir(FPATH_PREPROCESSED)]

# importing CSVs and joining into one dataframe
for index, df in enumerate(df_list):    
    if index == 0:
        data = pd.DataFrame(df)
    else:
        data = pd.merge(data, df, on="uid", how="left")

# TRAIN TEST SPLIT ------------------------------------------------------------
# there are a lot of targets with NAN, if we just use sklearns train_test_split
# there may be NANs in the test set which is not ideal.
# getting the indices where the targets are not NANs
not_nan_indices = data[np.logical_not(
        data.loc[:, "flourishing_scale_imp_post"].isnull())].index

# selecting (20% * total number of obs) at random from the targets that are not
# NANs
seed(RANDOM_STATE)
test_indices = sample(list(not_nan_indices), ceil(data.shape[0] * TEST_SIZE))

data_train = data.drop(test_indices, axis="rows")
data_test = data.iloc[test_indices, :]

# old code for train test split
#from sklearn.model_selection import train_test_split
#data_train, data_test = train_test_split(data,
#                                   test_size=TEST_SIZE,
#                                   random_state=RANDOM_STATE)

# just the features
X_train = data_train.drop(TARGET_LIST, axis="columns")
X_train.to_csv(os.path.join(FPATH_PROCESSED, "X_train.csv"))

X_test = data_test.drop(TARGET_LIST, axis="columns")
X_test.to_csv(os.path.join(FPATH_PROCESSED, "X_test.csv"))

# now the targets
for target in TARGET_LIST:
    # creating a df for each target
    exec("y_train_" + target + " = data_train.loc[:, ['uid', '" + target + "']]")
    # outputting to CSV
    exec("y_train_" + target + ".to_csv(os.path.join(FPATH_PROCESSED, 'y_train_"
                                                     + target + ".csv'))")    
    exec("y_test_" + target + " = data_test.loc[:, ['uid', '" + target + "']]")
    exec("y_test_" + target + ".to_csv(os.path.join(FPATH_PROCESSED, 'y_test_"
                                                     + target + ".csv'))")    

# clean up namespace
del data, data_test, data_train, df, df_list, index, target, test_indices
