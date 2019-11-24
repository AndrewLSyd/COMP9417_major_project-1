# -- coding: utf-8 --
"""
Created on 17/11/19
Useful functions for machine learning
@author: Andrew Lau
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.scorer import make_scorer
from matplotlib.ticker import FormatStrFormatter

from matplotlib.ticker import FormatStrFormatter

def quantile_plot(y_true, y_pred, quantiles=10, scale=1.0, title=None):
    """ 
    description: plots a quantile plot based on sorting observations by y_pred
        and grouping them into quantiles.
    inputs:
        y_true - array like object of actuals
        y_pred - array like object of predicted values
        quantiles - number of quantiles/bins to separate the data
        scale - the predicted values can be scaled to focus on the ability of
            the model to rank the observations
        title - plot title
    returns:
        DataFrame with y_true and y_pred grouped into quantiles sorted by y_pred
    """
    if scale == "auto":
        scale_used = sum(y_true) / sum(y_pred)
    else:
        scale_used = scale

    df = pd.DataFrame({'Actual':np.array(y_true).reshape(-1),
                                  'Predicted':np.array(y_pred).reshape(-1) * scale_used})
    df = df.groupby(pd.qcut(df.rank(method="first").loc[:, "Predicted"], quantiles)).mean()
    df.loc[:, "Quantile"] = (list(range(1, quantiles + 1)))
    df = df.set_index("Quantile")

    df.plot()
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d'))
    plt.gca().set_ylabel("Target")    
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(quantiles))
    
    if title:
        plt.title(title)    
    else:
        plt.title("Predicted vs. Observed (" + str(quantiles) + " quantiles)")

    return df


def gini_score(y_true, y_pre, **kwargs):
    """
    description: model scoring metric. returns gini index.
    input: y_true, y_pre
    returns: gini score
    adapted from: https://zhiyzuo.github.io/Plot-Lorenz/
    """
    sorted_arr = np.array([x for _, x in sorted(zip(y_true, y_pre))])
    n = sorted_arr.size
    coef_ = 2. / n
    const_ = (n + 1.) / n
    weighted_sum = sum([(i + 1) * yi for i, yi in enumerate(sorted_arr)])

    return coef_ * weighted_sum / (sorted_arr.sum()) - const_

gini_scorer = make_scorer(gini_score, greater_is_better=True)

def lorenz_curve(y_true, y_pred):
    """
    description: plots the lorenz curve based on y_true and y_pred.
    input: y_true, y_pre
    adapted from: https://zhiyzuo.github.io/Plot-Lorenz/
    """
    X = np.array([x for _,x in sorted(zip(y_true, y_pred))])

    X_lorenz = X.cumsum() / X.sum()
    X_lorenz = np.insert(X_lorenz, 0, 0)
    X_lorenz[0], X_lorenz[-1]
    fig, ax = plt.subplots(figsize=[6,6])
    ## scatter plot of Lorenz curve
    ax.scatter(np.arange(X_lorenz.size)/(X_lorenz.size-1), X_lorenz,
               marker='x', color='orange', s=5, label="Predicted Separation")

    # repeat for perfect separation
    X = np.array([x for _,x in sorted(zip(y_true, y_true))])

    X_lorenz = X.cumsum() / X.sum()
    X_lorenz = np.insert(X_lorenz, 0, 0)
    X_lorenz[0], X_lorenz[-1]

    ## scatter plot of Lorenz curve
    ax.scatter(np.arange(X_lorenz.size)/(X_lorenz.size-1), X_lorenz,
               marker='x', color='red', s=5, label="Perfect Separation")

    ## line plot of equality\
    ax.plot([0,1], [0,1], color='k')
    ax.set_title("Gini Index - Predicted: %.4f, Gini Index Perfect: %.4f" % (gini_score(y_true, y_pred), gini_score(y_true, y_true)))
    ax.legend()

def get_best_features_tree(tree_model, column_names):
    """ returns a pd.DataFrame with the best features of a sklearn tree based model """
    tree_model_feature_imp = pd.DataFrame({'feature':column_names, 'importance':tree_model.feature_importances_})
    tree_model_feature_imp = tree_model_feature_imp.sort_values('importance', ascending=False)
    return tree_model_feature_imp
