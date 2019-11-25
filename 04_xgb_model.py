import pandas as pd
from xgboost.sklearn import XGBClassifier, XGBRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import math
from sklearn.externals import joblib
import os
import shap
import helper_functions as helper

# Global variables needed
OUTPATH_RESULTS = "results/xgb/"
CV_FOLDS = 10
N_JOBS = 4
N_ITER = 80
SEED = 12
COLS_KEEP = ["rank_test_score", "mean_test_score", "std_test_score",
             "param_learning_rate", "param_gamma", "param_max_depth", "param_colsample_bytree", "param_subsample",
             "param_min_child_weight", "param_n_estimators", "param_reg_alpha", "param_reg_lambda"]

"""
If you have pre-trained the model, skip to the code section that retrieves the best model.
"""
X_train = pd.read_csv("train_test_data/X_train.csv", index_col="uid").drop("Unnamed: 0", axis="columns")
X_test = pd.read_csv("train_test_data/X_test.csv", index_col="uid").drop("Unnamed: 0", axis="columns")
y_test = pd.read_csv("train_test_data/y_test.csv", index_col="uid").drop("Unnamed: 0", axis="columns")
y_train = pd.read_csv("train_test_data/y_train.csv", index_col="uid").drop("Unnamed: 0", axis="columns")


"""
Retrieve our desired training data and test data
"""
# We only consider the post scores
class_names = [col for col in y_test.columns if "post" in col]
# Use the wk 9 to 10 training data only
used_weeks = [str(wk) for wk in range(9, 11)]
filtered_columns = X_train.columns[X_train.columns.str.endswith(tuple(used_weeks))]
# Select all wk 9 and 10 features
train_data = {"class": {}, "regr": {}}
test_data = {"class": {}, "regr": {}}
# Prepare training and test data for classification model and regression model respectively
for name in class_names:
    # Classification labels
    if "class" in name:
        target_type = "class"
    else:
        target_type = "regr"
    train_data[target_type][name] = {"x": {}, "y": {}}
    test_data[target_type][name] = {"x": {}, "y": {}}
    # Only consider data which has the post score being not null
    not_null_indices = np.logical_not(y_train.loc[:, name].isnull())
    train_data[target_type][name]["x"] = X_train[filtered_columns][not_null_indices]
    train_data[target_type][name]["y"] = y_train[not_null_indices].loc[:, name]
    test_data[target_type][name]["x"] = X_test[filtered_columns]
    test_data[target_type][name]["y"] = y_test.loc[:, name]


# Classifier parameter grid
class_parameters = {"learning_rate": [0.001, 0.01, 0.1], 
                    "gamma" : [0, 0.01, 0.05, 0.1],
                    "max_depth": [1, 3, 5, 7],
                    "colsample_bytree": [0.3, 0.5, 0.7, 1.0],
                    "subsample": [0.3, 0.5, 0.7, 1.0],
                    "min_child_weight": [1, 2, 3],
                    "n_estimators": [100, 300, 500, 1000, 1500],
                    "reg_alpha": [0],
                    "reg_lambda": [1]
                    }


# Function to find the best hyperparameters for each model
def find_best_model(rscv, x, y):
    best_model = rscv.fit(x, y)
    return best_model


def report_classifier_performance(model, x, y):
    predict = model.predict(x)
    predict_proba = model.predict_proba(x)[:, 1]
    print("Accuracy :", metrics.accuracy_score(y, predict))
    print("logloss:", metrics.log_loss(y, predict_proba))
    print("AUC Score:", metrics.roc_auc_score(y, predict_proba))
    print()


# from https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# Baseline classifier
baseline_classifier = XGBClassifier(objective='binary:logistic', scale_pos_weight=1)
rscv_classifier = RandomizedSearchCV(baseline_classifier, class_parameters,
                                     scoring="neg_log_loss", verbose=False, cv=CV_FOLDS,
                                     n_jobs=N_JOBS, n_iter=N_ITER, iid=False, random_state=SEED)
# Find the optimal classifier and its performance on training and test set respectively
for name in train_data["class"]:
    print("Prediction of " + name)
    best_model = find_best_model(rscv_classifier, train_data["class"][name]["x"], train_data["class"][name]["y"])
    # Save the best model in pickle
    joblib.dump(best_model, 'xgb_model_' + name)
    report(best_model.cv_results_)
    print("Performance on training set:")
    report_classifier_performance(best_model, train_data["class"][name]["x"], train_data["class"][name]["y"])
    print("Performance on test set:")
    report_classifier_performance(best_model, test_data["class"][name]["x"], test_data["class"][name]["y"])
    print("-------------------------------------------------------------------------------------------------\n")


def report_regression_performance(model, x, y):
    predict = model.predict(x)
    print("MSE :", metrics.mean_squared_error(y, predict))
    print("RMSE :", math.sqrt(metrics.mean_squared_error(y, predict)))
    df = pd.DataFrame({'Actual': np.array(y).reshape(-1),
                       'Predicted': np.array(predict).reshape(-1)})
    print()
    return df


# Classifier parameter grid
regr_parameters = {"learning_rate": [0.001, 0.01, 0.1], 
                    "gamma" : [0, 0.01, 0.05, 0.1],
                    "max_depth": [1, 3, 5, 7],
                    "colsample_bytree": [0.3, 0.5, 0.7, 1.0],
                    "subsample": [0.3, 0.5, 0.7, 1.0],
                    "min_child_weight": [1, 2, 3],
                    "n_estimators": [100, 300, 500, 1000, 1500],
                    'reg_alpha': [1e-5, 1e-2,  0.75],
                    'reg_lambda': [1e-5, 1e-2, 0.45],
                   }


# Baseline regressor
baseline_regressor = XGBRegressor(objective="reg:squarederror")
xgb_rscv_regressor = RandomizedSearchCV(baseline_regressor, param_distributions=regr_parameters, 
                                        scoring="neg_mean_squared_error", iid=False, random_state=12,
                                        cv=CV_FOLDS, verbose=False, n_jobs=N_JOBS, n_iter=N_ITER)

# Find the optimal classifier and its performance on training and test set respectively
for name in train_data["regr"]:
    print("Prediction of " + name)
    best_model = find_best_model(xgb_rscv_regressor, train_data["regr"][name]["x"], train_data["regr"][name]["y"])
    # Save the best model in pickle
    joblib.dump(best_model, 'xgb_model_' + name)
    report(best_model.cv_results_)
    print("Performance on training set:")
    report_regression_performance(best_model, train_data["regr"][name]["x"], train_data["regr"][name]["y"])
    print("Performance on test set:")
    report_regression_performance(best_model, test_data["regr"][name]["x"], test_data["regr"][name]["y"])
    print("-------------------------------------------------------------------------------------------------\n")


# Retrieve best models
flourishing_scale_imp_class_post = joblib.load('xgb_model_flourishing_scale_imp_class_post')
flourishing_scale_raw_class_post = joblib.load('xgb_model_flourishing_scale_raw_class_post')
flourishing_scale_imp_post = joblib.load('xgb_model_flourishing_scale_imp_post')
flourishing_scale_raw_post = joblib.load('xgb_model_flourishing_scale_raw_post')
panas_neg_imp_post = joblib.load('xgb_model_panas_neg_imp_post')
panas_neg_raw_post = joblib.load('xgb_model_panas_neg_raw_post')
panas_pos_imp_post = joblib.load('xgb_model_panas_pos_imp_post')
panas_pos_raw_post = joblib.load('xgb_model_panas_pos_raw_post')
panas_neg_imp_class_post = joblib.load('xgb_model_panas_neg_imp_class_post')
panas_neg_raw_class_post = joblib.load('xgb_model_panas_neg_raw_class_post')
panas_pos_imp_class_post = joblib.load('xgb_model_panas_pos_imp_class_post')
panas_pos_raw_class_post = joblib.load('xgb_model_panas_pos_raw_class_post')


# Used to retrieve the grid search results
def combine_imp_raw_results(raw_cv_results, imp_cv_results, fpath_out):
    """
        takes the cv results from the randomised grid search object from the raw and imputed target models
        combines them, sorts by score and outputs
        """
    df_raw = pd.DataFrame(raw_cv_results)[COLS_KEEP]
    df_raw.insert(3, "param__target_imputation", "No Imputation")

    df_imp = pd.DataFrame(imp_cv_results)[COLS_KEEP]
    df_imp.insert(3, "param__target_imputation", "KNN Imputation")

    df = pd.concat([df_raw, df_imp]).sort_values("mean_test_score", ascending=False)
    df.rank_test_score = list(range(1, df.shape[0] + 1))
    df.columns = ["Rank", "CV score (mean)", "CV score (standard deviation)", 
                  "Target imputation", "Learning rate", "Gamma", 
                  "Max depth", "Max features in each tree (split)", "Subsample", 
                  "Min child weight", "Number of estimators (boost rounds)", "Alpha", "Lambda"]
    df["CV score (mean)"] = -df["CV score (mean)"]
    df.to_csv(fpath_out)
    return df


if not os.path.isdir(OUTPATH_RESULTS):
    os.mkdir(OUTPATH_RESULTS)
combine_imp_raw_results(flourishing_scale_raw_class_post.cv_results_,
                        flourishing_scale_imp_class_post.cv_results_,
                        OUTPATH_RESULTS + "flourishing_scale_class_post" + ".csv")
combine_imp_raw_results(flourishing_scale_raw_post.cv_results_,
                        flourishing_scale_imp_post.cv_results_,
                        OUTPATH_RESULTS + "flourishing_scale_post" + ".csv")

combine_imp_raw_results(panas_pos_raw_class_post.cv_results_,
                        panas_pos_imp_class_post.cv_results_,
                        OUTPATH_RESULTS + "panas_pos_class_post" + ".csv")
combine_imp_raw_results(panas_pos_raw_post.cv_results_,
                        panas_pos_imp_post.cv_results_,
                        OUTPATH_RESULTS + "panas_pos_post" + ".csv")
combine_imp_raw_results(panas_neg_raw_class_post.cv_results_,
                        panas_neg_imp_class_post.cv_results_,
                        OUTPATH_RESULTS + "panas_neg_class_post" + ".csv")
combine_imp_raw_results(panas_neg_raw_post.cv_results_,
                        panas_neg_imp_post.cv_results_,
                        OUTPATH_RESULTS + "panas_neg_post" + ".csv")


# functions to output results
def output_results_df(cv_results, cols_keep, target):    
    pd.DataFrame(cv_results).sort_values("rank_test_score", ascending=True)[cols_keep].to_csv(OUTPATH_RESULTS + target + ".csv")


def output_results_diagnostics(estimator, target, X_train, y_train, X_test, y_test, quantiles=5, classifier=False, metric_name="Score"):
    y_train_pred = estimator.predict(X_train)
    y_test_pred = estimator.predict(X_test)
    if classifier:
        y_train_pred_proba = estimator.predict_proba(X_train)[:,1]
        y_test_pred_proba = estimator.predict_proba(X_test)[:,1]  
        score_train = metrics.log_loss(y_train, y_train_pred_proba)
        score_test = metrics.log_loss(y_test, y_test_pred_proba)
        metric_name = "Log Loss"
        helper.lorenz_curve(y_train, y_train_pred_proba)
        plt.savefig(OUTPATH_RESULTS + target + "_lorenz_train.png")
        helper.lorenz_curve(y_test, y_test_pred_proba)   
        plt.savefig(OUTPATH_RESULTS + target + "_lorenz_test.png")
    else:
        score_train = metrics.mean_squared_error(y_train, y_train_pred)
        score_test = metrics.mean_squared_error(y_test, y_test_pred)      
        metric_name = "MSE"
        helper.lorenz_curve(y_train, y_train_pred)
        plt.savefig(OUTPATH_RESULTS + target + "_lorenz_train.png")
        helper.lorenz_curve(y_test, y_test_pred)   
        plt.savefig(OUTPATH_RESULTS + target + "_lorenz_test.png")

    helper.quantile_plot(y_train, y_train_pred, quantiles=quantiles, title=metric_name + " train: {:.4f}".format(score_train))
    plt.savefig(OUTPATH_RESULTS + target + "_pvo_train.png")

    helper.quantile_plot(y_test, y_test_pred, quantiles=quantiles, title=metric_name + " test: {:.4f}".format(score_test))
    plt.savefig(OUTPATH_RESULTS + target + "_pvo_test.png")
        

def output_results_shap(estimator, target, X, corr_thresh=0.3):
    shap_values = shap.TreeExplainer(estimator).shap_values(X)
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=10, color="orange")
    plt.savefig(OUTPATH_RESULTS + target + "_shap_bar.png", bbox_inches='tight')
    shap.summary_plot(shap_values, X, show=False, max_display=10)
    plt.savefig(OUTPATH_RESULTS + target + "_shap.png", bbox_inches='tight')


def output_results(estimator, cv_results, target, X_train, y_train, X_test, y_test, cols_keep, quantiles, classifier=False,
                  metric_name="Score"):
    plt.clf()
    output_results_df(cv_results, cols_keep, target)
    plt.clf()
    output_results_diagnostics(estimator, target, X_train, y_train, X_test, y_test, classifier=classifier, quantiles=quantiles, metric_name=metric_name)
    plt.clf()
    output_results_shap(estimator.best_estimator_, target, X_train)
    plt.clf()


output_results(flourishing_scale_raw_class_post, flourishing_scale_raw_class_post.cv_results_, 
               "flourishing_scale_raw_class_post", train_data["class"]["flourishing_scale_raw_class_post"]["x"],
               train_data["class"]["flourishing_scale_raw_class_post"]["y"],
               test_data["class"]["flourishing_scale_raw_class_post"]["x"],
               test_data["class"]["flourishing_scale_raw_class_post"]["y"], COLS_KEEP, 5, classifier=True)

output_results(flourishing_scale_raw_post, flourishing_scale_raw_post.cv_results_, 
               "flourishing_scale_raw_post", train_data["regr"]["flourishing_scale_raw_post"]["x"],
               train_data["regr"]["flourishing_scale_raw_post"]["y"],
               test_data["regr"]["flourishing_scale_raw_post"]["x"],
               test_data["regr"]["flourishing_scale_raw_post"]["y"], COLS_KEEP, 5, classifier=False)

output_results(panas_pos_raw_class_post, panas_pos_raw_class_post.cv_results_, 
               "panas_pos_raw_class_post", train_data["class"]["panas_pos_raw_class_post"]["x"],
               train_data["class"]["panas_pos_raw_class_post"]["y"],
               test_data["class"]["panas_pos_raw_class_post"]["x"],
               test_data["class"]["panas_pos_raw_class_post"]["y"], COLS_KEEP, 5, classifier=True)

output_results(panas_pos_raw_post, panas_pos_raw_post.cv_results_, 
               "panas_pos_raw_post", train_data["regr"]["panas_pos_raw_post"]["x"],
               train_data["regr"]["panas_pos_raw_post"]["y"],
               test_data["regr"]["panas_pos_raw_post"]["x"],
               test_data["regr"]["panas_pos_raw_post"]["y"], COLS_KEEP, 5, classifier=False)

output_results(panas_neg_raw_class_post, panas_neg_raw_class_post.cv_results_, 
               "panas_neg_raw_class_post", train_data["class"]["panas_neg_raw_class_post"]["x"],
               train_data["class"]["panas_neg_raw_class_post"]["y"],
               test_data["class"]["panas_neg_raw_class_post"]["x"],
               test_data["class"]["panas_neg_raw_class_post"]["y"], COLS_KEEP, 5, classifier=True)

output_results(panas_neg_raw_post, panas_neg_raw_post.cv_results_, 
               "panas_neg_raw_post", train_data["regr"]["panas_neg_raw_post"]["x"],
               train_data["regr"]["panas_neg_raw_post"]["y"],
               test_data["regr"]["panas_neg_raw_post"]["x"],
               test_data["regr"]["panas_neg_raw_post"]["y"], COLS_KEEP, 5, classifier=False)

