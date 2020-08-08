# generic imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve

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

def lin_model(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
    residuals = y_test - yhat
    return model, yhat, residuals

def plot_resid_fitted(yhat, residuals):
    '''
    input is predicted y and residuals
    returns residual plot
    '''
    fig, ax = plt.subplots(1,1, figsize=(10,8))
    ax.scatter(yhat, residuals)
    ax.axhline(0, linestyle ='--')
    #ax.set_xlabel('y predicted HIV diagnoses', size=16)
    ax.set_ylabel('residuals (y observed - y predicted)', size=16)
    ax.set_title('residual plot', size=18)
    #ax.set_ylim((-200,500))
    #plt.savefig('../images/residuals.png')
    return fig, ax

def get_stats(X_test, y_true, y_pred):
    '''
    inputs X, y actual and y predicted
    returns useful stats for linear models
    '''
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred) 
    r2 = metrics.r2_score(y_true, y_pred)
    # train.shape[0] is n and # train.shape[1] is p (features or predictors)
    adj_r2 = (1 - (1 - r2) * ((X_test.shape[0] - 1) / 
          (X_test.shape[0] - X_test.shape[1] - 1)))
    print(f'explained_variance: {round(explained_variance, 4)}') 
    print(f'r2: {round(r2,4)}')
    print(f'adj. r2: {round(adj_r2,4)}')
    print(f'MSE: {round(mse,4)}')
    print(f'RMSE: {round(np.sqrt(mse),4)}') # in units of data
    return explained_variance, mse, r2, adj_r2

def get_coeffs(model, X_test, cols):
    '''
    inputs regression model and pd df with feature names as cols
    returns dict of coeffs
    '''
    coef_dict = {}
    coeffs = model.coef_[0].tolist() 
    intercept = model.intercept_
    # put in dict
    for coef, feat in zip(coeffs, cols):
        coef_dict[feat] = round(coef, 4)
    # print
    [print(f'{k} : {v}') for k, v in coef_dict.items()]
    print(f'The intercept is: {np.round(intercept, 3)}')
    return coef_dict, intercept

def grad_boost(X_train, X_test, y_train, y_test):
    
    model = GradientBoostingClassifier(loss='deviance',
                             n_estimators=1000,
                             learning_rate=0.1,
                             max_depth=3,
                             subsample=0.5,
                             min_samples_split=2,
                             min_samples_leaf=1,
                             min_weight_fraction_leaf=0.0)
    model.fit(X_train, y_train.values.ravel()) # needs a label not y_train???
    yhat = model.predict_proba(X_test)
    score = model.score(X_test, y_test)
    #gbmodel.predict_proba(X_test)
    return model, yhat, score

def decision_tree(X_train, X_test, y_train):
    '''
    Creates decision tree model
    '''
    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    return y_pred, score

# def plot_decision_tree(X_train, y_train):
#     '''
#     Creates plot for decision tree boundaries
#     '''
#     clf = tree.DecisionTreeClassifier()
#     clf = clf.fit(X_train, y_train)
#     tree.plot_tree(clf) 

def plot_classification_tree(ax, X, y, model=None, fit = True):
    ax.plot(X[y==0, 0], X[y==0, 1], 'r.', label='')
    ax.plot(X[y==1, 0], X[y==1, 1], 'b.', label='')
    ax.set_title("Classifying Active Status With Decision Trees")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(loc='upper left')
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plot_classification_thresholds(ax, X, y, 0, model.tree_, xlim, ylim)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

def Random_Forest(X_train, X_test, y_train):
    '''
    Creates Random Forest model
    '''
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    return y_pred, score

def feature_importance(model, names):

    feature_importances = 100*model.feature_importances_ / np.sum(model.feature_importances_)
    feature_importances, feature_names, feature_idxs = \
    zip(*sorted(zip(feature_importances, names, range(len(names)))))

    width = 0.8

    idx = np.arange(len(names))
    plt.barh(idx, feature_importances, align='center')
    plt.yticks(idx, feature_names)

    plt.title("Feature Importances")
    plt.xlabel('Relative Importance of Feature', fontsize=14)
    plt.ylabel('Feature Name', fontsize=14)
    
#### MEGANS ####
'''

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble.partial_dependence import plot_partial_dependence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df = pd.read_csv('../data/cleaned_churn_train.csv')

%matplotlib inline

import textwrap # for wrapping answer descriptions


'''
import textwrap # for wrapping answer descriptions

def loopallgridsearch(models, params, X_train, y_train):
    results = []
    for model, param in zip(models, params):
        
        # run grid search
        gridsearch = GridSearchCV(model,
                                param,
                                n_jobs=-1,
                                verbose=True)
                                #scoring='neg_mean_squared_error')
        # fit best model
        gridsearch.fit(X_train, y_train)
        # get best model parameters
        print(f'"best parameters:", {gridsearch.best_params_}')
        # return best model parameters
        best_model = gridsearch.best_estimator_
        results.append(best_model)
    return results
    #print(GridSearchCV(model, return_train_score=True))


def GridSearch_(X_train, y_train):
    # params dict
    grad_boost_grid = {'learning_rate': [0.1],
                   'loss': ['deviance'],
                   'subsample': [0.3, 0.5, 0.7],
                      'n_estimators': [10, 20, 40, 80, 100, 300],
                   'max_depth': [2,3],
                   #'min_samples_leaf': [7, 9, 13],
                   'max_features': [3, 5, 10, 15],
                      'random_state': [1]}
    gdb_gridsearch = GridSearchCV(GradientBoostingClassifier(),
                                grad_boost_grid,
                                n_jobs=-1,
                                verbose=True)
                                #scoring='neg_mean_squared_error')
    gdb_gridsearch.fit(X_train, y_train)
    print(f'"best parameters:", {gdb_gridsearch.best_params_}')
    best_gdb_model = gdb_gridsearch.best_estimator_
    return best_gdb_model


if __name__ == '__main__':
    #data = pd.read_csv('../data/cleaned_churn_train.csv')
    # can create X and y but lets use Charle's subsets
    X_train = pd.read_csv('../data/Cleaned_X_train.csv')
    X_test = pd.read_csv('../data/Cleaned_X_test.csv')
    y_train = pd.read_csv('../data/Cleaned_y_train.csv')
    y_test = pd.read_csv('../data/Cleaned_y_test.csv')
    #X_train = X_train
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
    #GridSearch_(X_train_std, y_train_ravel)


    #models = [DecisionTreeClassifier, GradientBoostClassifier, RandomForestClassifier, AdaBoostClassifier]
    models = [GradientBoostingClassifier(), RandomForestClassifier(), DecisionTreeClassifier()]
  
    # define param dicts for each model
    grad_boost_grid = {'learning_rate': [0.001, 0.01, 0.1],
                    'loss': ['deviance'],
                    'subsample': [0.3, 0.5, 0.7],
                        #'n_estimators': [10, 20, 40, 80, 100, 300],
                        'n_estimators': [10, 100, 1000],
                    'max_depth': [2, 3],
                    #'min_samples_leaf': [7, 9, 13],
                    #'max_features': [3, 10],
                    'max_features': ['auto'],
                        'random_state': [1]}
                        
    random_forest_grid = {
                    'criterion': ['gini', 'entropy'],
                    'n_estimators': [50, 100, 300, 500],
                    'max_depth': [2,3],
                    'random_state': [1]}

    decision_tree_grid = {
                    'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random'],
                    'max_depth': [2,3],
                    'max_features': [3, 5, 10, 15],
                    'random_state': [1]}


    paramslst = [grad_boost_grid, random_forest_grid, decision_tree_grid]
    
    # for model, param in zip(models, paramslst):
    #     print(model, param)
    lst_models = loopallgridsearch(models, paramslst, X_train, y_train_ravel)
    # lst_models[0].score(X_test, y_test) 
    # lst_models[1].score(X_test, y_test) 
    # lst_models[2].score(X_test, y_test) 
    # gdb = GradientBoostingClassifier(
    #                          n_estimators=1000,
    #                          learning_rate=0.001,
    #                          max_depth=2,
    #                          subsample=0.3,
    #                          min_samples_split=2,
    #                          min_samples_leaf=1,
    #                          min_weight_fraction_leaf=0.0)
    # gdb.fit(X_train, y_train_ravel)
    # print(gdb.score(X_test, y_test))

    # rf = RandomForestClassifier(
    #                 criterion='gini',
    #                 n_estimators=100,
    #                 max_depth=2,
    #                 n_jobs=-1)
    # rf.fit(X_train, y_train_ravel)
    # print(rf.score(X_test, y_test))

    dt = DecisionTreeClassifier(
                    criterion='gini',
                    max_depth=2,
                    max_features=10)
    dt.fit(X_train, y_train_ravel)
    print(dt.score(X_test, y_test))

    #feature_importance(gdb, cols)
    #feature_importance(rf, cols)
    feature_importance(dt, cols)
    plt.show()
