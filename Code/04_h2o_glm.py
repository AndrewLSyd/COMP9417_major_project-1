import os
import sys
import pickle
from copy import deepcopy
from collections import defaultdict

import numpy as np
import pandas as pd

import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.grid.grid_search import H2OGridSearch
import helper_functions as helper

# start a h2o instance to fit GLM models
# pick settings so things run fast but don't use all system resources
h2o.init(nthreads = 3, max_mem_size = "12G")


def import_data(data):
    return pd.read_csv("train_test_data/" + data + ".csv", index_col="uid").drop("Unnamed: 0", axis="columns")


X_train = import_data("X_train")
X_test = import_data("X_test")
y_train = import_data("y_train")
y_test = import_data("y_test")

y_train = y_train[[e for e in y_train.columns if 'post' in e]]
y_test = y_test[[e for e in y_test.columns  if 'post' in e]]
train_cols = X_train.columns.tolist()

# remove rows where targets all all NA
train = pd.concat([y_train, X_train], axis = 1)
train = train.dropna(how = 'all', subset = ['panas_pos_imp_post', 'panas_neg_imp_post', 'panas_pos_imp_post'])
test = pd.concat([y_test, X_test], axis = 1)
test = test.dropna(how = 'all', subset = ['panas_pos_imp_post', 'panas_neg_imp_post', 'panas_pos_imp_post'])

# h2o likes to convert mostly na values into categories. so
# we copy the pandas type mapping across
col_types = dict(train.dtypes)
replacements = {'float64': 'real',
                'int64': 'int'}
for e in col_types:
    col_types[e] = replacements[str(col_types[e])]

train_h2o = h2o.H2OFrame(train, column_types=col_types)
test_h2o = h2o.H2OFrame(test, column_types=col_types)

all_models = dict()
best_models = dict()
# restrict search to only imputated cases where imputation actually occurs
targets = ['flourishing_scale_raw_class_post',
           'flourishing_scale_raw_post',
           'panas_neg_raw_class_post',
           'panas_neg_raw_post',
           'panas_pos_raw_class_post',
           'panas_pos_raw_post',
           'panas_pos_imp_class_post',
           'panas_pos_imp_post',
           ]
for target in targets:
    print(f"searching for best model for target {target}")
    if 'class' in target:
        families = ['binomial']
        metric_name = 'logloss'
    else:
        # TODO: tweedie_variance_power and tweedie_link_power (for tweedie) to work
        families = ["gaussian", "tweedie", "gamma", "poisson", "negativebinomial"]
        metric_name = 'mse'
    output_models = defaultdict(pd.DataFrame)
    best_metic_value = np.Inf
    best_family = None
    best_model = None
    for features in ['all', 'wk_10', 'wk_9-10']:
        if features == 'all':
            x_cols = train_cols
        elif features == 'wk_10':
            x_cols = [e for e in train_cols if 'wk_10' in e]
        elif features == 'wk_9-10':
            x_cols = [e for e in train_cols if 'wk_9' in e or 'wk_10' in e]
        else:
            raise ValueError('feature set not encoded')
        for family in families:
            print(f"searching for best model in {family} family")
            hyper_parameters = {'alpha': list(np.arange(0, 1.1, 0.1))}

            # h2o grid search doesn't support searching tweedie distribution over the
            # space of canonical link functions so we define a custom search to support
            # this
            if family == "tweedie":
                # define a simple space (noting that both Guassian, Poisson and Gamma)
                # are already covered in other cases
                tweedie_variance_powers = [1.1, 1.3, 1.5, 1.7, 1.9]
            else:
                tweedie_variance_powers = [0]
            if family == "negativebinomial":
                hyper_parameters['theta'] = [1e-10, 1e-8, 1e-4, 1e-2, 0.1, 0.5, 1]

            for vp in tweedie_variance_powers:
                h2o_glm = H2OGeneralizedLinearEstimator(family=family, nfolds=5, seed=20191106,
                                                        # tweedie parameters are ignored if not tweedie distn.
                                                        tweedie_variance_power=vp,
                                                        tweedie_link_power=1.0 - vp)
                gs = H2OGridSearch(h2o_glm, hyper_parameters)

                gs.train(y=target, x=x_cols, training_frame=train_h2o)
                glm_grid_models = gs.get_grid(sort_by='mse')

                num_models = len(list(glm_grid_models.get_grid()))

                model_results = {
                    'response': target,
                    'family': family,
                    'alpha': [glm_grid_models.get_hyperparams(e)[0] for e in range(num_models)],
                    'best_lambda': [e.actual_params['lambda'][0] for e in glm_grid_models],
                    'metric_name': metric_name,
                    'features': features
                }
                if 'class' in target:
                    model_results['metric_value'] = list(
                        glm_grid_models.get_grid(sort_by="mse").logloss(xval=True).values())
                else:
                    model_results['metric_value'] = list(
                        glm_grid_models.get_grid(sort_by="mse").mse(xval=True).values())

                if family == "tweedie":
                    model_results['tweedie_power'] = vp
                elif family == "negativebinomial":
                    model_results['theta'] = [glm_grid_models.get_hyperparams(e)[1] for e in range(num_models)]
                # keep track of all models
                output_models[family] = output_models[family].append(pd.DataFrame(model_results), ignore_index=True)

                family_best_model = glm_grid_models.models[0]

                if 'class' in target:
                    if family_best_model.logloss(xval=True) < best_metic_value:
                        print(f"!! Classification new best model is {family} with {features} features !!")
                        print(f"old value {best_metic_value}, new value {family_best_model.logloss(xval=True)}")
                        best_model = family_best_model
                        best_metic_value = family_best_model.logloss(xval=True)
                        best_family = family
                else:
                    if family_best_model.mse(xval=True) < best_metic_value:
                        print(f"!! Regression new best model is {family} with {features} features !!")
                        print(f"old value {best_metic_value}, new value {family_best_model.mse(xval=True)}")
                        best_model = family_best_model
                        best_metic_value = family_best_model.mse(xval=True)
                        best_family = family
    all_models[target] = deepcopy(output_models)
    h2o.save_model(model=best_model, path=f"./fitted_models/h2o_glm/{target}", force=True)
    best_models[target] = {'best_model': best_model,
                           'metric_value': metric_name,
                           'best_metic_value': best_metic_value,
                           'best_family': best_family,
                           'features': 'features'}

with open('fitted_models/h2o_glm/best_models.pkl', 'wb') as out_file:
    pickle.dump(best_models, out_file, protocol=pickle.HIGHEST_PROTOCOL)

all_models_pd = dict()
for e in all_models:
    all_models_pd[e] = pd.concat(all_models[e], ignore_index = True)
all_models_pd = pd.concat(all_models_pd, ignore_index = True, sort=False)
all_models_pd.sort_values(by=['response', 'metric_value'], inplace = True)
all_models_pd.to_csv("./fitted_models/h2o_glm/glm_cv_results.csv")
all_models_pd

for model in best_models:
    print(f'--------------- {model} ---------------')
    print(best_models[model]['best_model'])

# make predictions for all models
all_predictions = dict()
for model in best_models:
    all_predictions[model] = best_models[model]['best_model'].predict(test_h2o)

# large number of models are just the constant model
# that is using linear regression we can't beat a straight line
# without further feature engineering
# this makes sense (since non-linear behaviour is expected)
for model in best_models:
    print(f'--------------- {model} ---------------')
    print(all_predictions[model])

for target in best_models:
    print(f'--------------- {target} ---------------')
    print(best_models[target]['best_model'].model_performance(test_data=test_h2o))

all_coefs = list()
for target in best_models:
    all_coefs.append(
        pd.DataFrame.from_dict(best_models[target]['best_model'].coef(),
                               orient='index',
                               columns=[target])
    )

all_coefs = pd.concat(all_coefs, axis=1)
all_coefs.to_csv("./fitted_models/h2o_glm/glm_coefs.csv")
all_coefs

# TO DO create summary metrics and stuff
# only need to compare hyper-parameters for raw vs imputated at a high level

# get top 10 parameters per model
# auc curve
# pvo
# confusion matrix