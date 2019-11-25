import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import log_loss, mean_squared_error
import shap
import helper_functions as helper
from matplotlib import pyplot as plt
import warnings
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
# warnings.filterwarnings(action='once')
warnings.filterwarnings('ignore')


def import_data(data):    
    return pd.read_csv("train_test_data/" + data + ".csv", index_col="uid").drop("Unnamed: 0", axis="columns")        


X_train = import_data("X_train")
X_test = import_data("X_test")
y_train = import_data("y_train")
y_test = import_data("y_test")

X_train.head()
X_test.head()
y_train.head()
y_test.head()


X_train.head()
X_train.describe()


y_train.describe()


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


X_train.iloc[:,features_wk_9_10_ind]
np.array(X_train).shape


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


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names="all"):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.attribute_names == "all":
            return X            
        if self.attribute_names == "wk_9_10":            
            return np.array(X)[:,features_wk_9_10_ind]            
        if self.attribute_names == "wk_10":
            return np.array(X)[:,features_wk_10_ind]    


def grid_search(estimator, target, scoring, nn=25, n_iter=1000, cv=10, n_jobs=-1, n_top=1, verbose=True):
    
    print("*" * 20, target, "*" * 20)
    df_selector = DataFrameSelector()
    imputer = SimpleImputer()
    scaler = StandardScaler()
    pca = PCA()
    pipe = Pipeline(steps=[('df_selector', df_selector), ('imputer', imputer), ('scaler', scaler), ('pca', pca), 
                           ('knn', estimator)])
    
    param_grid = {
        "df_selector__attribute_names": ["all", "wk_9_10", "wk_10"],
        "imputer__strategy": ["most_frequent", "mean", "median"],
        "pca__n_components": [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99],
        "knn__n_neighbors": list(range(1,nn)),
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"]
    }
    
    # train using only labelled data
    target_not_null = np.logical_not(y_train.loc[:, target].isnull())
    
    # run randomised search
    random_search = RandomizedSearchCV(pipe, param_distributions=param_grid, n_iter=n_iter, cv=cv, n_jobs=n_jobs,
                                       scoring=scoring, iid=False, random_state=SEED)
    random_search.fit(X_train[target_not_null], y_train[target_not_null].loc[:, target])
    
    if verbose:
        report(random_search.cv_results_, n_top=n_top)
    
    return random_search


def f(row):
    if row['Actual'] == row['Predicted']:
        return 1.0
    else:
        return 0.0


def tabulate(y_actual, y_pred):
    df = pd.DataFrame({'Actual':np.array(y_actual).reshape(-1), 'Predicted':np.array(y_pred).reshape(-1)})

    df['Correct'] = df.apply(f, axis=1)
    return df


def output_results_df(cv_results, cols_keep, target):    
    pd.DataFrame(cv_results).sort_values("rank_test_score", ascending=True)[cols_keep].to_csv(OUTPATH_RESULTS + target + ".csv")


def output_results_diagnostics(estimator, target, X_train, y_train, X_test, y_test, quantiles=5, classifier=False):
    target_train_not_null = np.logical_not(y_train.loc[:, target].isnull())
    y_train_actual = y_train[target_train_not_null].loc[:, target]
    y_train_pred = estimator.predict(X_train[target_train_not_null])

    target_test_not_null = np.logical_not(y_test.loc[:, target].isnull())
    y_test_actual = y_test[target_test_not_null].loc[:, target]
    y_test_pred = estimator.predict(X_test[target_test_not_null])
    
    if classifier:
        y_train_pred_proba = estimator.predict_proba(X_train[target_train_not_null])[:,1]
        y_test_pred_proba = estimator.predict_proba(X_test[target_test_not_null])[:,1]  
        score_train = -log_loss(y_train_actual, y_train_pred_proba)
        score_test = -log_loss(y_test_actual, y_test_pred_proba)
    else:
        score_train = mean_squared_error(y_train_actual, y_train_pred)
        score_test = mean_squared_error(y_test_actual, y_test_pred)
        
#     plt.clf()
    helper.quantile_plot(y_train_actual, y_train_pred, quantiles=quantiles, title="Score train: {:.4f}".format(score_train))
    plt.savefig(OUTPATH_RESULTS + target + "_pvo_train.png")
#     plt.clf()
    helper.quantile_plot(y_test_actual, y_test_pred, quantiles=quantiles, title="Score test: {:.4f}".format(score_test))
    plt.savefig(OUTPATH_RESULTS + target + "_pvo_test.png")
#     plt.clf()
    if classifier:
        helper.lorenz_curve(y_train_actual, y_train_pred_proba)
        plt.savefig(OUTPATH_RESULTS + target + "_lorenz_train.png")
        helper.lorenz_curve(y_test_actual, y_test_pred_proba)   
        plt.savefig(OUTPATH_RESULTS + target + "_lorenz_test.png")
#         plt.clf()
    else:
        helper.lorenz_curve(y_train_actual, y_train_pred)
        plt.savefig(OUTPATH_RESULTS + target + "_lorenz_train.png")
        helper.lorenz_curve(y_test_actual, y_test_pred)   
        plt.savefig(OUTPATH_RESULTS + target + "_lorenz_test.png")
#         plt.clf()


def output_results_shap(estimator, target, X, corr_thresh=0.3):
    shap_values = shap.KernelExplainer(estimator.predict, X).shap_values(X)
#     plt.clf()
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False, max_display=10)
    plt.savefig(OUTPATH_RESULTS + target + "_shap_bar.png", bbox_inches='tight')
#     plt.clf()
    shap.summary_plot(shap_values, X_train, show=False, max_display=10)
    plt.savefig(OUTPATH_RESULTS + target + "_shap.png", bbox_inches='tight')
#     plt.clf()


def output_results(estimator, cv_results, target, X_train, y_train, X_test, y_test, cols_keep, quantiles, classifier=False):
    plt.clf()
    output_results_df(cv_results, cols_keep, target)
    plt.clf()
    output_results_diagnostics(estimator, target, X_train, y_train, X_test, y_test, classifier=classifier, quantiles=quantiles)
    plt.clf()
    output_results_shap(estimator, target, X_train)
    plt.clf()


knn_flourishing_scale_imp_class_post = grid_search(KNeighborsClassifier(algorithm="auto"),
                                     "flourishing_scale_imp_class_post", "neg_log_loss")


pred_tar = "flourishing_scale_imp_class_post"
tnn_train = np.logical_not(y_train.loc[:, pred_tar].isnull())
tnn_test = np.logical_not(y_test.loc[:, pred_tar].isnull())

y_actual = np.array(y_test[tnn_test].loc[:, pred_tar])
y_pred = knn_flourishing_scale_imp_class_post.best_estimator_.predict_proba(X_test[tnn_test])

log_loss(y_actual, y_pred)


knn_flourishing_scale_imp_post = grid_search(KNeighborsRegressor(algorithm="auto"),
                                     "flourishing_scale_imp_post", "neg_mean_squared_error")


pred_tar = "flourishing_scale_imp_post"
tnn_train = np.logical_not(y_train.loc[:, pred_tar].isnull())
tnn_test = np.logical_not(y_test.loc[:, pred_tar].isnull())

y_actual = np.array(y_test[tnn_test].loc[:, pred_tar])
y_pred = knn_flourishing_scale_imp_post.best_estimator_.predict(X_test[tnn_test])

mean_squared_error(y_actual, y_pred)


knn_panas_pos_imp_class_post = grid_search(KNeighborsClassifier(algorithm="auto"),
                                          "panas_pos_imp_class_post", "neg_log_loss")


pred_tar = "panas_pos_imp_class_post"
tnn_train = np.logical_not(y_train.loc[:, pred_tar].isnull())
tnn_test = np.logical_not(y_test.loc[:, pred_tar].isnull())

y_actual = np.array(y_test[tnn_test].loc[:, pred_tar])
y_pred = knn_panas_pos_imp_class_post.best_estimator_.predict_proba(X_test[tnn_test])

-log_loss(y_actual, y_pred)


knn_panas_pos_imp_post = grid_search(KNeighborsRegressor(algorithm="auto"),
                                     "panas_pos_imp_post", "neg_mean_squared_error")


pred_tar = "panas_pos_imp_post"
tnn_train = np.logical_not(y_train.loc[:, pred_tar].isnull())
tnn_test = np.logical_not(y_test.loc[:, pred_tar].isnull())

y_actual = np.array(y_test[tnn_test].loc[:, pred_tar])
y_pred = knn_panas_pos_imp_post.best_estimator_.predict(X_test[tnn_test])

mean_squared_error(y_actual, y_pred)


knn_panas_neg_imp_class_post = grid_search(KNeighborsClassifier(algorithm="auto"),
                                          "panas_neg_imp_class_post", "neg_log_loss")


pred_tar = "panas_neg_imp_class_post"
tnn_train = np.logical_not(y_train.loc[:, pred_tar].isnull())
tnn_test = np.logical_not(y_test.loc[:, pred_tar].isnull())

y_actual = np.array(y_test[tnn_test].loc[:, pred_tar])
y_pred = knn_panas_neg_imp_class_post.best_estimator_.predict_proba(X_test[tnn_test])

-log_loss(y_actual, y_pred)

knn_panas_neg_imp_post = grid_search(KNeighborsRegressor(algorithm="auto"),
                                     "panas_neg_imp_post", "neg_mean_squared_error")


pred_tar = "panas_neg_imp_post"
tnn_train = np.logical_not(y_train.loc[:, pred_tar].isnull())
tnn_test = np.logical_not(y_test.loc[:, pred_tar].isnull())

y_actual = np.array(y_test[tnn_test].loc[:, pred_tar])
y_pred = knn_panas_neg_imp_post.best_estimator_.predict(X_test[tnn_test])

mean_squared_error(y_actual, y_pred)


OUTPATH_RESULTS = "results/knn/"
COLS_KEEP = ['rank_test_score', 'mean_test_score', 'std_test_score', 'param_pca__n_components', 'param_imputer__strategy',
       'param_df_selector__attribute_names', 'param_knn__n_neighbors', 'param_knn__weights', 'param_knn__metric']

output_results(knn_flourishing_scale_imp_class_post, knn_flourishing_scale_imp_class_post.cv_results_, 
               "flourishing_scale_imp_class_post", X_train, y_train, X_test, y_test, COLS_KEEP, 5, classifier=True)
output_results(knn_flourishing_scale_imp_post , knn_flourishing_scale_imp_post.cv_results_,
               "flourishing_scale_imp_post", X_train, y_train, X_test, y_test, COLS_KEEP, 5)
output_results(knn_panas_pos_imp_class_post , knn_panas_pos_imp_class_post.cv_results_,
               "panas_pos_imp_class_post", X_train, y_train, X_test, y_test, COLS_KEEP, 5, classifier=True)
output_results(knn_panas_pos_imp_post, knn_panas_pos_imp_post.cv_results_,
               "panas_pos_imp_post", X_train, y_train, X_test, y_test, COLS_KEEP, 5)
output_results(knn_panas_neg_imp_class_post , knn_panas_neg_imp_class_post.cv_results_,
               "panas_neg_imp_class_post", X_train, y_train, X_test, y_test, COLS_KEEP, 5, classifier=True)
output_results(knn_panas_neg_imp_post, knn_panas_neg_imp_post.cv_results_,
               "panas_neg_imp_post", X_train, y_train, X_test, y_test, COLS_KEEP, 5)







