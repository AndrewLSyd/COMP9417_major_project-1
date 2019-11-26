# import modules
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn 
print (sklearn.__version__)
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.svm import SVC, SVR
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
import helper_functions as helper  # importing custom helper functions
import shap
import pickle
import os

# globals
CV_FOLDS = 10
N_ITER = 10_000
N_JOBS = 8
# whilst the results may not converge, further iterations yield little benefit compared to testing out another hyper-parameter
MAX_ITER = 5000

# import data
def import_data(data):    
    return pd.read_csv("train_test_data/" + data + ".csv", index_col="uid").drop("Unnamed: 0", axis="columns")


X_train = import_data("X_train")
X_test = import_data("X_test")
y_train = import_data("y_train")
y_test = import_data("y_test")

# checking import
X_train.head()
X_test.head()
y_train.head()
y_test.head()


y_train.describe()
y_train.columns

# histograms of the target for EDA
y_train.loc[:, ['flourishing_scale_imp_class_pre']].hist()
y_train.loc[:, ['flourishing_scale_raw_post']].hist()


y_train.loc[:, ['panas_neg_raw_pre']].hist()
y_train.loc[:, ['panas_neg_raw_post']].hist()


y_train.loc[:, ['panas_pos_raw_post']].hist()
y_train.loc[:, ['panas_pos_raw_pre']].hist()

# feature subset lists
features_wk_10 = [
    "chargetime_count_wk_10"
    , "chargetime_max_wk_10"
    , "chargetime_mean_wk_10"
    , "chargetime_median_wk_10"
    , "chargetime_min_wk_10"
    , "chargetime_q1_wk_10"
    , "chargetime_q3_wk_10"
    , "activity_stationary_ratio_wk_10"
    , "activity_running_ratio_wk_10"
    , "audio_silent_ratio_wk_10"
    , "audio_noisy_ratio_wk_10"
    , "bluetooth_avg_wk_10"
    , "conversation_hours_wk_10"
    , "conversation_freq_wk_10"
    , "speed_mean_wk_10"
    , "speed_max_wk_10"
    , "speed_sd_wk_10"
    , "travelstate_time_stationary_wk_10"
    , "travelstate_time_moving_wk_10"
    , "outdoor_time_wk_10"
    , "indoor_time_wk_10"
    , "indoor_dist_wk_10"
    , "outdoors_dist_wk_10"
    , "altitude_mean_wk_10"
    , "altitude_sd_wk_10"
    , "altitude_max_wk_10"
    , "altitude_min_wk_10"
    , "location_count_wk_10"
    , "location_1_time_wk_10"
    , "location_2_time_wk_10"
    , "location_3_time_wk_10"
    , "location_4_time_wk_10"
    , "location_5_time_wk_10"
    , "bearing_north_time_wk_10"
    , "bearing_east_time_wk_10"
    , "bearing_south_time_wk_10"
    , "bearing_west_time_wk_10"
    , "sleep_max_wk_10"
    , "sleep_mean_wk_10"
    , "sleep_med_wk_10"
    , "sleep_min_wk_10"
    , "locktime_count_wk_10"
    , "locktime_max_wk_10"
    , "locktime_mean_wk_10"
    , "locktime_median_wk_10"
    , "locktime_min_wk_10"
    , "locktime_q1_wk_10"
    , "locktime_q3_wk_10"
]
features_wk_9_10 = [
    "chargetime_count_wk_9"
    , "chargetime_max_wk_9"
    , "chargetime_mean_wk_9"
    , "chargetime_median_wk_9"
    , "chargetime_min_wk_9"
    , "chargetime_q1_wk_9"
    , "chargetime_q3_wk_9"
    , "activity_stationary_ratio_wk_9"
    , "activity_running_ratio_wk_9"
    , "audio_silent_ratio_wk_9"
    , "audio_noisy_ratio_wk_9"
    , "bluetooth_avg_wk_9"
    , "conversation_hours_wk_9"
    , "conversation_freq_wk_9"
    , "speed_mean_wk_9"
    , "speed_max_wk_9"
    , "speed_sd_wk_9"
    , "travelstate_time_stationary_wk_9"
    , "travelstate_time_moving_wk_9"
    , "outdoor_time_wk_9"
    , "indoor_time_wk_9"
    , "indoor_dist_wk_9"
    , "outdoors_dist_wk_9"
    , "altitude_mean_wk_9"
    , "altitude_sd_wk_9"
    , "altitude_max_wk_9"
    , "altitude_min_wk_9"
    , "location_count_wk_9"
    , "location_1_time_wk_9"
    , "location_2_time_wk_9"
    , "location_3_time_wk_9"
    , "location_4_time_wk_9"
    , "location_5_time_wk_9"
    , "bearing_north_time_wk_9"
    , "bearing_east_time_wk_9"
    , "bearing_south_time_wk_9"
    , "bearing_west_time_wk_9"
    , "sleep_max_wk_9"
    , "sleep_mean_wk_9"
    , "sleep_med_wk_9"
    , "sleep_min_wk_9"
    , "locktime_count_wk_9"
    , "locktime_max_wk_9"
    , "locktime_mean_wk_9"
    , "locktime_median_wk_9"
    , "locktime_min_wk_9"
    , "locktime_q1_wk_9"
    , "locktime_q3_wk_9"
]
features_wk_9_10.extend(features_wk_10)


features_wk_9_10_ind = [X_train.columns.get_loc(c) for c in features_wk_9_10]
features_wk_10_ind = [X_train.columns.get_loc(c) for c in features_wk_10]


# from https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html
def report(results, n_top=3):
    """
    print the results of the CV randomised grid search
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Allows selection of subsets of features in a SKLearn Pipeline object.
    Adapted from "Hands-On Machine learning with Sciki-Learn and TensorFlow by Geron"
    TransformerMixIn that allows selection of features.
    """
    def __init__(self, attribute_names="all"):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # allows selection of different feature subsets
        if self.attribute_names == "all":
            return X            
        if self.attribute_names == "wk_9_10":            
            return np.array(X)[:,features_wk_9_10_ind]            
        if self.attribute_names == "wk_10":
            return np.array(X)[:,features_wk_10_ind]            


def grid_search(estimator, param_grid, target, scoring, n_iter=50, cv=10, n_jobs=8, n_top=3, verbose=True):
    """
    run a randomised grid search on a target
    """
    print("*" * 20, target, "*" * 20)
    df_selector = DataFrameSelector()
    imputer = SimpleImputer()
    scaler = StandardScaler()
    pca = PCA()
    pipe = Pipeline(steps=[('df_selector', df_selector), ('imputer', imputer), ('scaler', scaler),
                           ('pca', pca), ('SVM', estimator)])

    # train using only labelled data (toss out null values)
    target_not_null = np.logical_not(y_train.loc[:, target].isnull())

    # run randomized search
    search = RandomizedSearchCV(pipe, param_grid,
                                       n_iter=n_iter, 
                          cv=cv, iid=False, n_jobs=n_jobs, scoring=scoring)
    search.fit(X_train[target_not_null], y_train[target_not_null].loc[:, target])

    # print results
    if verbose:
        report(search.cv_results_, n_top=n_top) 
    return search  

# tuning grid for random grid search
param_grid = {
    "df_selector__attribute_names":["all", "wk_9_10", "wk_10"],
    "imputer__strategy":["most_frequent", "mean", "median"],    
    "pca__n_components": [0.6, 0.7, 0.8, 0.85, 0.9, 0.95],  # proportion of total variation
    "SVM__C": [0.01, 0.1, 1, 10, 100, 1000],
    "SVM__kernel": ["linear", "rbf", "poly"],
    "SVM__gamma": [0.01, 0.1, 1, 10, 100],
    "SVM__degree": [2, 3, 4, 5, 6]
}

# grid search for flourishing - imputed
flourishing_scale_imp_class_post = grid_search(SVC(max_iter=MAX_ITER, probability=True), param_grid,
                                     "flourishing_scale_imp_class_post", "neg_log_loss", n_jobs=N_JOBS, n_iter=N_ITER)
flourishing_scale_imp_post = grid_search(SVR(max_iter=MAX_ITER), param_grid,
                               "flourishing_scale_imp_post", "neg_mean_squared_error", n_jobs=N_JOBS, n_iter=N_ITER)
pickle.dump(flourishing_scale_imp_class_post, open('results/SVM/flourishing_scale_imp_class_post.sklearnmodel', 'wb'))
pickle.dump(flourishing_scale_imp_post, open('results/SVM/flourishing_scale_imp_post.sklearnmodel', 'wb'))


# grid search for flourishing - raw
flourishing_scale_raw_class_post = grid_search(SVC(max_iter=MAX_ITER, probability=True), param_grid,
                                     "flourishing_scale_raw_class_post", "neg_log_loss", n_jobs=N_JOBS, n_iter=N_ITER)
flourishing_scale_raw_post = grid_search(SVR(max_iter=MAX_ITER), param_grid,
                               "flourishing_scale_raw_post", "neg_mean_squared_error", n_jobs=N_JOBS, n_iter=N_ITER)
pickle.dump(flourishing_scale_raw_class_post, open('results/SVM/flourishing_scale_raw_class_post.sklearnmodel', 'wb'))
pickle.dump(flourishing_scale_raw_post, open('results/SVM/flourishing_scale_raw_post.sklearnmodel', 'wb'))


# grid search for panas - imp
panas_pos_imp_class_post = grid_search(SVC(max_iter=MAX_ITER, probability=True), param_grid,
                                     "panas_pos_imp_class_post", "neg_log_loss", n_jobs=N_JOBS, n_iter=N_ITER)
panas_pos_imp_post = grid_search(SVR(max_iter=MAX_ITER), param_grid,
                               "panas_pos_imp_post", "neg_mean_squared_error", n_jobs=N_JOBS, n_iter=N_ITER)
panas_neg_imp_class_post = grid_search(SVC(max_iter=MAX_ITER, probability=True), param_grid,
                                     "panas_neg_imp_class_post", "neg_log_loss", n_jobs=N_JOBS, n_iter=N_ITER)
panas_neg_imp_post = grid_search(SVR(max_iter=MAX_ITER), param_grid,
                               "panas_neg_imp_post", "neg_mean_squared_error", n_jobs=N_JOBS, n_iter=N_ITER)
pickle.dump(panas_pos_imp_class_post, open('results/SVM/panas_pos_imp_class_post.sklearnmodel', 'wb'))
pickle.dump(panas_pos_imp_post, open('results/SVM/panas_pos_imp_post.sklearnmodel', 'wb'))
pickle.dump(panas_neg_imp_class_post, open('results/SVM/panas_neg_imp_class_post.sklearnmodel', 'wb'))
pickle.dump(panas_neg_imp_post, open('results/SVM/panas_neg_imp_post.sklearnmodel', 'wb'))


# grid search for panas
panas_pos_raw_class_post = grid_search(SVC(max_iter=MAX_ITER, probability=True), param_grid,
                                     "panas_pos_raw_class_post", "neg_log_loss", n_jobs=N_JOBS, n_iter=N_ITER)
panas_pos_raw_post = grid_search(SVR(max_iter=MAX_ITER), param_grid,
                               "panas_pos_raw_post", "neg_mean_squared_error", n_jobs=N_JOBS, n_iter=N_ITER)
panas_neg_raw_class_post = grid_search(SVC(max_iter=MAX_ITER, probability=True), param_grid,
                                     "panas_neg_raw_class_post", "neg_log_loss", n_jobs=N_JOBS, n_iter=N_ITER)
panas_neg_raw_post = grid_search(SVR(max_iter=MAX_ITER), param_grid,
                               "panas_neg_raw_post", "neg_mean_squared_error", n_jobs=N_JOBS, n_iter=N_ITER)
pickle.dump(panas_pos_raw_class_post, open('results/SVM/panas_pos_raw_class_post.sklearnmodel', 'wb'))
pickle.dump(panas_pos_raw_post, open('results/SVM/panas_pos_raw_post.sklearnmodel', 'wb'))
pickle.dump(panas_neg_raw_class_post, open('results/SVM/panas_neg_raw_class_post.sklearnmodel', 'wb'))
pickle.dump(panas_neg_raw_post, open('results/SVM/panas_neg_raw_post.sklearnmodel', 'wb'))


# load pickled models - imputed
flourishing_scale_imp_class_post = pickle.load(open('results/SVM/flourishing_scale_imp_class_post.sklearnmodel', 'rb'))
flourishing_scale_imp_post = pickle.load(open('results/SVM/flourishing_scale_imp_post.sklearnmodel', 'rb'))
panas_pos_imp_class_post = pickle.load(open('results/SVM/panas_pos_imp_class_post.sklearnmodel', 'rb'))
panas_pos_imp_post = pickle.load(open('results/SVM/panas_pos_imp_post.sklearnmodel', 'rb'))
panas_neg_imp_class_post = pickle.load(open('results/SVM/panas_neg_imp_class_post.sklearnmodel', 'rb'))
panas_neg_imp_post = pickle.load(open('results/SVM/panas_neg_imp_post.sklearnmodel', 'rb'))


# load pickled models - raw
flourishing_scale_raw_class_post = pickle.load(open('results/SVM/flourishing_scale_raw_class_post.sklearnmodel', 'rb'))
flourishing_scale_raw_post = pickle.load(open('results/SVM/flourishing_scale_raw_post.sklearnmodel', 'rb'))
panas_pos_raw_class_post = pickle.load(open('results/SVM/panas_pos_raw_class_post.sklearnmodel', 'rb'))
panas_pos_raw_post = pickle.load(open('results/SVM/panas_pos_raw_post.sklearnmodel', 'rb'))
panas_neg_raw_class_post = pickle.load(open('results/SVM/panas_neg_raw_class_post.sklearnmodel', 'rb'))
panas_neg_raw_post = pickle.load(open('results/SVM/panas_neg_raw_post.sklearnmodel', 'rb'))


# functions to output results
def output_results_df(cv_results, cols_keep, target):
    """
    takes in the cv results from a grid search cv object, creates and outputs a dataframe
    """
    pd.DataFrame(cv_results).sort_values("rank_test_score", ascending=True)[cols_keep].to_csv(OUTPATH_RESULTS + target + ".csv")


def output_results_diagnostics(estimator, target, X_train, y_train, X_test, y_test, quantiles=5, classifier=False, metric_name="Score"):
    """
    run the PvO and Lorenz curve diagnostics for an estimator
    """
    # only use non-null target values
    target_train_not_null = np.logical_not(y_train.loc[:, target].isnull())
    y_train_actual = y_train[target_train_not_null].loc[:, target]
    y_train_pred = estimator.predict(X_train[target_train_not_null])

    target_test_not_null = np.logical_not(y_test.loc[:, target].isnull())
    y_test_actual = y_test[target_test_not_null].loc[:, target]
    y_test_pred = estimator.predict(X_test[target_test_not_null])
    
    if classifier:
        # classifier uses predicted probability
        y_train_pred_proba = estimator.predict_proba(X_train[target_train_not_null])[:,1]
        y_test_pred_proba = estimator.predict_proba(X_test[target_test_not_null])[:,1]  
        score_train = log_loss(y_train_actual, y_train_pred_proba)
        score_test = log_loss(y_test_actual, y_test_pred_proba)
        metric_name = "Log Loss"
    else:
        score_train = mean_squared_error(y_train_actual, y_train_pred)
        score_test = mean_squared_error(y_test_actual, y_test_pred)      
        metric_name = "MSE"
    # PvO by quantile plot
    helper.quantile_plot(y_train_actual, y_train_pred, quantiles=quantiles, title=metric_name + " train: {:.4f}".format(score_train))
    plt.savefig(OUTPATH_RESULTS + target + "_pvo_train.png")

    helper.quantile_plot(y_test_actual, y_test_pred, quantiles=quantiles, title=metric_name + " test: {:.4f}".format(score_test))
    plt.savefig(OUTPATH_RESULTS + target + "_pvo_test.png")

    if classifier:
        helper.lorenz_curve(y_train_actual, y_train_pred_proba)
        plt.savefig(OUTPATH_RESULTS + target + "_lorenz_train.png")
        helper.lorenz_curve(y_test_actual, y_test_pred_proba)   
        plt.savefig(OUTPATH_RESULTS + target + "_lorenz_test.png")
    else:
        helper.lorenz_curve(y_train_actual, y_train_pred)
        plt.savefig(OUTPATH_RESULTS + target + "_lorenz_train.png")
        helper.lorenz_curve(y_test_actual, y_test_pred)   
        plt.savefig(OUTPATH_RESULTS + target + "_lorenz_test.png")


def output_results_shap(estimator, target, X, corr_thresh=0.3):
    """
    Produce SHAP diagnostics for an estimator
    """
    shap_values = shap.KernelExplainer(estimator.predict, X).shap_values(X)
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False, max_display=10, color="orange")
    plt.savefig(OUTPATH_RESULTS + target + "_shap_bar.png", bbox_inches='tight')
    shap.summary_plot(shap_values, X_train, show=False, max_display=10)
    plt.savefig(OUTPATH_RESULTS + target + "_shap.png", bbox_inches='tight')


def output_results(estimator, cv_results, target, X_train, y_train, X_test, y_test, cols_keep, quantiles, classifier=False,
                  metric_name="Score"):
    """
    run all diagnostics and results outputting functions
    """
    plt.clf()
    output_results_df(cv_results, cols_keep, target)
    plt.clf()
    output_results_diagnostics(estimator, target, X_train, y_train, X_test, y_test, classifier=classifier, quantiles=quantiles,
                              metric_name=metric_name)
    plt.clf()
    output_results_shap(estimator, target, X_train)
    plt.clf()


# produce outputs
OUTPATH_RESULTS = "results/SVM/"
COLS_KEEP = ['rank_test_score', 'mean_test_score', 'std_test_score',
             'param_pca__n_components', 'param_imputer__strategy',
             'param_df_selector__attribute_names', 'param_SVM__kernel',
             'param_SVM__gamma', 'param_SVM__degree', 'param_SVM__C']

output_results(flourishing_scale_imp_class_post, flourishing_scale_imp_class_post.cv_results_, 
               "flourishing_scale_imp_class_post", X_train, y_train, X_test, y_test, COLS_KEEP, 5, classifier=True)
output_results(flourishing_scale_imp_post , flourishing_scale_imp_post.cv_results_,
               "flourishing_scale_imp_post", X_train, y_train, X_test, y_test, COLS_KEEP, 5)
output_results(panas_pos_imp_class_post , panas_pos_imp_class_post.cv_results_,
               "panas_pos_imp_class_post", X_train, y_train, X_test, y_test, COLS_KEEP, 5, classifier=True)
output_results(panas_pos_imp_post, panas_pos_imp_post.cv_results_,
               "panas_pos_imp_post", X_train, y_train, X_test, y_test, COLS_KEEP, 5)
output_results(panas_neg_imp_class_post , panas_neg_imp_class_post.cv_results_,
               "panas_neg_imp_class_post", X_train, y_train, X_test, y_test, COLS_KEEP, 5, classifier=True)
output_results(panas_neg_imp_post, panas_neg_imp_post.cv_results_,
               "panas_neg_imp_post", X_train, y_train, X_test, y_test, COLS_KEEP, 5)


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
    
    df = df.rename(columns={"rank_test_score":"Rank", "mean_test_score":"CV score (mean)",
                            "std_test_score":"CV score (standard deviation)",
                            "param__target_imputation":"Target imputation",
                            "param_pca__n_components":"PCA prop. var.",
                            "param_imputer__strategy":"Imputer strategy",
                            "param_df_selector__attribute_names":"Feature subset",
                            "param_SVM__kernel":"SVM kernel",
                            "param_SVM__gamma":"SVM gamma", "param_SVM__degree":"SVM degree",
                            "param_SVM__C":"SVM C",})
    df["CV score (mean)"] = -1.0 * df["CV score (mean)"]
    
    df.to_csv(fpath_out)
    return df


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

