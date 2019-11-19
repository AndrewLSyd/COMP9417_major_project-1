# -- coding: utf-8 --
"""
Created on 17/11/19
Useful functions for machine learning
@author: AndreL01
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.scorer import make_scorer

def quantile_plot(y_actual, y_pred, quantiles=10, scale=1.0):
    """ 
    purpose: plots a decile plot based on deciling by y_pred. returns the df with
    the deciles.
    inputs:
        y_actual - array like object of actuals
        y_pred - array like object of predicted values
    returns:
        DataFrame with quantiles
    """
    if scale == "auto":
        scale_used = sum(y_actual) / sum(y_pred)
    else:
        scale_used = scale

    df = pd.DataFrame({'Actual':np.array(y_actual).reshape(-1),
                                  'Predicted':np.array(y_pred).reshape(-1) * scale_used})
    df = df.groupby(pd.qcut(df.rank(method="first").loc[:,"Predicted"] , quantiles)).mean()
    df.loc[:, "quantile"] = (list(range(quantiles)))
    df = df.set_index("quantile")

    df.plot()
    plt.xticks(rotation=90)
    plt.title("Predicted vs. Observed (" + str(quantiles) + " quantiles)")

    return df

def gini_score(y_true, y_pre, **kwargs):
    """
    description: model scoring metric. returns gini index.
    input: y_true, y_pre
    adapted from: https://zhiyzuo.github.io/Plot-Lorenz/
    """
    sorted_arr = np.array([x for _,x in sorted(zip(y_true, y_pre))])
    n = sorted_arr.size
    coef_ = 2. / n
    const_ = (n + 1.) / n
    weighted_sum = sum([(i + 1) * yi for i, yi in enumerate(sorted_arr)])

    return coef_*weighted_sum/(sorted_arr.sum()) - const_

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

def shap_feature_imp(df_shap,df, corr_thresh=0.3):
    """
    creates a shap plot of feature importance with bars coloured indicated
    direction of effect. Red = positively correlated with target, Blue = negatively
    correlated with target
    inputs:
        df_shap - shap object
        df - dataframe with features
        corr_thresh - the threshold at which we include the feature
    adapted from
    https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d
    """
    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    #    df_v = df.copy().reset_index().drop('index',axis=1)
    df_v = df.copy().reset_index()
    
    # Determine the correlation in order to plot with different colors
    corr_list = []
    feature_list_selected = []
    for i in feature_list:
        b = np.corrcoef(shap_v[i],df_v[i])[1][0]
        if abs(b) > corr_thresh:
            corr_list.append(b)
            feature_list_selected.append(i)
    
    corr_df = pd.concat([pd.Series(feature_list_selected),pd.Series(corr_list)],axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns  = ['Variable','Corr']
    corr_df['Sign'] = np.where(corr_df['Corr']>0,'red','blue')
    
    # Plot it
    shap_abs = np.abs(shap_v)
    k=pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable','SHAP_abs']
    k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
    k2 = k2.sort_values(by='SHAP_abs',ascending = True)
    colorlist = k2['Sign']
    ax = k2.plot.barh(x='Variable',y='SHAP_abs',color = colorlist, figsize=(5,6),legend=False)
    ax.set_xlabel("SHAP Value (Red = Positive Impact)")