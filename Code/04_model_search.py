"""
perform a model search with default ML methods on the data to get an indication as to
which models may be worth investigating as well as provide a benchmark.
"""
# globals
CV_FOLDS = 10
N_JOBS = 1


# import modules
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import sklearn 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# import data
def import_data(data):    
    return pd.read_csv("train_test_data/" + data + ".csv", index_col="uid").drop("Unnamed: 0", axis="columns")        
X_train = import_data("X_train")
X_test = import_data("X_test")
y_train = import_data("y_train")
y_test = import_data("y_test")

# check import
X_train.head()
X_test.head()
y_train.head()
y_test.head()

y_train.describe()

y_train.columns

y_train.loc[:, ['flourishing_scale_imp_class_pre']].hist()
y_train.loc[:, ['flourishing_scale_raw_post']].hist()

y_train.loc[:, ['panas_neg_raw_pre']].hist()
y_train.loc[:, ['panas_neg_raw_post']].hist()

y_train.loc[:, ['panas_pos_raw_post']].hist()
y_train.loc[:, ['panas_pos_raw_pre']].hist()

imputer = SimpleImputer(strategy="most_frequent")
scaler = StandardScaler()

X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)

X_train.head()
X_train.describe()


# model list classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

models_clf = [RandomForestClassifier(n_estimators=100), KNeighborsClassifier(), SVC(gamma="auto", probability=True), GaussianProcessClassifier(),
          DecisionTreeClassifier(), AdaBoostClassifier(n_estimators=100), GaussianNB(),
          QuadraticDiscriminantAnalysis(), MLPClassifier(), XGBClassifier(), LogisticRegression(solver="lbfgs")]


# model list regression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

models_reg = [RandomForestRegressor(n_estimators=100), KNeighborsRegressor(), SVR(gamma="auto"), GaussianProcessRegressor(),
          DecisionTreeRegressor(), AdaBoostRegressor(n_estimators=100), MLPRegressor(max_iter=5000), XGBRegressor(objective="reg:squarederror")]


def model_search(model_list, X, y, target, scoring, bigger_score_is_better=True, cv_folds=CV_FOLDS, n_jobs=N_JOBS):
    """
    searches across a grid of models
    """
    model_results = []

    target_not_null = np.logical_not(y_train.loc[:, target].isnull())

    for model in model_list:
        scores = cross_val_score(estimator=model,
                                 X=X_train[target_not_null],
                                 y=y_train[target_not_null].loc[:, target], 
                                 scoring=scoring,
                                 cv=cv_folds, 
                                 n_jobs=n_jobs)

        model_results.append([str(model.__class__), np.mean(scores), np.std(scores)])
    
    return pd.DataFrame(model_results, columns=['model_' + target, 'CV_score_' + scoring, 'std']).sort_values('CV_score_' + scoring, ascending=not bigger_score_is_better)

# classification
model_search_flour_class_pre = model_search(models_clf, X_train, y_train, "flourishing_scale_imp_class_pre", "neg_log_loss")
model_search_flour_class_post = model_search(models_clf, X_train, y_train, "flourishing_scale_imp_class_post", "neg_log_loss")

# regression
model_search_flour_pre = model_search(models_reg, X_train, y_train, "flourishing_scale_imp_pre", "neg_mean_squared_error")
model_search_flour_post = model_search(models_reg, X_train, y_train, "flourishing_scale_imp_post", "neg_mean_squared_error")

# model_search_flour_class_pre
model_search_flour_class_post
# model_search_flour_pre
model_search_flour_post

# panas regression
model_search_panas_pos_pre = model_search(models_reg, X_train, y_train, "panas_pos_imp_pre", "neg_mean_squared_error")
model_search_panas_pos_post = model_search(models_reg, X_train, y_train, "panas_pos_imp_post", "neg_mean_squared_error")
model_search_panas_neg_pre = model_search(models_reg, X_train, y_train, "panas_neg_imp_pre", "neg_mean_squared_error")
model_search_panas_neg_post = model_search(models_reg, X_train, y_train, "panas_neg_imp_post", "neg_mean_squared_error")

# panas regression
model_search_panas_pos_class_pre = model_search(models_clf, X_train, y_train, "panas_pos_imp_class_pre", "neg_mean_squared_error")
model_search_panas_pos_class_post = model_search(models_clf, X_train, y_train, "panas_pos_imp_class_post", "neg_mean_squared_error")
model_search_panas_neg_class_pre = model_search(models_clf, X_train, y_train, "panas_neg_imp_class_pre", "neg_mean_squared_error")
model_search_panas_neg_class_post = model_search(models_clf, X_train, y_train, "panas_neg_imp_class_post", "neg_mean_squared_error")

# model_search_panas_pos_pre
model_search_panas_pos_post
# model_search_panas_neg_pre
model_search_panas_neg_post

# model_search_panas_pos_class_pre
model_search_panas_pos_class_post
# model_search_panas_neg_class_pre
model_search_panas_neg_class_post