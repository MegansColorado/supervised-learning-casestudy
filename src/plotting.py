# generic imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

# other settings
pd.options.display.float_format = '{:,.4f}'.format
pd.set_option("display.precision", 3)
np.set_printoptions(precision=3, suppress=True)

# sklearn stuff
from sklearn import tree
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV, cross_validate, cross_val_score
from sklearn.linear_model import LinearRegression, LassoLarsIC, Ridge, RidgeCV, Lasso, LassoCV, ElasticNetCV
from sklearn import metrics, datasets
from scipy.stats import probplot
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.metrics import roc_curve

def create_roc_curve(X_train, X_test, y_train, y_test, estimator, ax):
    # yea but I think it needs to be like axs or something
    #nevermind.. haha cuz what I was doing would just make subplots.. but almost like we want to do what we did by mistake earlier on purpose lol
    #i think if you put the estimators in a list, it should work

    #fig, ax = plt.subplots()
    for i in estimator:
        clf = i.fit(X_train, y_train)
        metrics.plot_roc_curve(i, X_test, y_test, ax=ax)
    #plt.legend() #or would this be ax.legend?
    #plt.show()
    return ax
    # return roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)

def standardize(X, method):
    # this Fit to data, then transform it.
    # Standard scaler -3 to 3 (better if data is normal)
    # also known a Z scale, unit variance, normal
    # this enables you to do the same on the train data and test data
    if method == 'ss':
        print('returning std scale')
        return StandardScaler().fit_transform(X)
    # Min max scalar 0 to 1 (better if not normal data)
    # doesn't handle outliers well
    # but preserves zeros in data
    if method == 'minmax':
        print('returning minmax scale')
        return MinMaxScaler().fit_transform(X)

if __name__ == '__main__':
    #data = pd.read_csv('../data/cleaned_churn_train.csv')
    # can create X and y but lets use Charle's subsets
    X_train = pd.read_csv('../data/Cleaned_X_train.csv')
    X_test = pd.read_csv('../data/Cleaned_X_test.csv')
    y_train = pd.read_csv('../data/Cleaned_y_train.csv')
    y_test = pd.read_csv('../data/Cleaned_y_test.csv')
    # create features list
    cols = list(X_train.columns)
    # standardize
    X_train_std = standardize(X_train, 'ss')
    X_test_std = standardize(X_test, 'ss')
    # print(X_train)
    # linear model with everything
    #model, yhat, residuals = lin_model(X_train_std, X_test_std, y_train, y_test)
    #print(yhat)
    # get stats
    #get_stats(X_test_std, y_test, yhat)
    # get coeffs
    #get_coeffs(model, X_test_std, cols)
    # plot resids
    #plot_resid_fitted(yhat, residuals)
    #plt.show()
    # but our y is categorical not continous...
    # so need to try these models:
    # 1. random forest
    # 2. gradient boosted
    #gbmodel, yhat, gbscore = grad_boost(X_train_std, X_test_std, y_train, y_test)
    # fix y train
    y_train_ravel = y_train.values.ravel()
    fig, ax = plt.subplots(1, figsize=(10, 8))
    lst = [rf, dt, gdb]
    create_roc_curve(X_train, X_test, y_train_ravel, y_test, lst, ax)  
    #create_roc_curve(X_train, X_test, y_train_ravel, y_test, dt, ax) 
    #create_roc_curve(X_train, X_test, y_train_ravel, y_test, gdb, ax)  
    #plt.legend()
    plt.show()