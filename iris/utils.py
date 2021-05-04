import numpy as np

#modules for Regression models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

#models for Classification models
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#modules for classification metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

#module for grid search
from sklearn.model_selection import GridSearchCV

#module for writing into csv
import csv


def summarize_data(df, categorical, continuos, predicted_column, type = 'regression'):
    print('\n\nPredicted Column: %s'%predicted_column)
    if type == 'regression':
        print(df[predicted_column].describe())
        classes = -1
    elif type == 'classification':
        print(df[predicted_column].value_counts())
        classes = len(df[predicted_column].unique())
    print('\n')
    for column in categorical:
        print('Predictor Column: %s'%column)
        print(df[column].value_counts())
        print('\n')
    for column in continuos:
        print('Predictor Column: %s'%column)
        print(df[column].describe())
        print('\n')
    return classes


def print_metrics(y, yhat, model, name, first = False):

    print('\t r2: %.3f'%r2_score(y, yhat))
    print('\t rmse: %.3f'%np.sqrt(mean_squared_error(y, yhat)))
    print('\t mae: %.3f'%mean_absolute_error(y, yhat))
    
    row = [model, round(r2_score(y, yhat), 2), round(np.sqrt(mean_squared_error(y, yhat)), 2)
                   , round(mean_absolute_error(y, yhat), 2)]
    
    file_name = name + '_metric.csv'
    
    if first:
        mode = 'w'
    else:
        mode = 'a'
    
    with open(file_name, mode) as file:
        writer = csv.writer(file)
        if first:
            header = ['model', 'r2', 'rmse', 'mae']
            writer.writerow(header)
        writer.writerow(row)

def fit_regression_models(train_X, train_y, test_X, test_y, name):
    #Linear Regression
    en_model = ElasticNet(max_iter = 10)
    en__model = GridSearchCV(en_model, param_grid = {'alpha': [0, 0.25, 0.5], 'l1_ratio':[0, 0.25, 0.5]})
    en__model.fit(train_X, train_y)
    predictions = en__model.predict(test_X)
    
    #printing metrics and writing to file
    print('Linear Regression')
    print("\t Best alpha: %.2f"%en__model.best_params_['alpha'])
    print("\t Best l1_ratio: %.2f"%en__model.best_params_['l1_ratio'])
    print_metrics(test_y, predictions, 'Linear Regression', name, first = True)
    


    #Decision Tree Regression
    dt_model = DecisionTreeRegressor(random_state = 7)
    dt__model = GridSearchCV(dt_model, param_grid = {'max_depth': [5, 10, 15]}, cv = 3)
    dt__model.fit(train_X, train_y)
    predictions = dt__model.predict(test_X)
    
    #printing metrics and writing to file
    print('Decision Tree')
    print("\t Best max_depth: %d"%dt__model.best_params_['max_depth'])
    print_metrics(test_y, predictions, 'Decision Tree', name)

    #Random Forest Regression
    rf_model = RandomForestRegressor(random_state = 7)
    rf__model = GridSearchCV(rf_model, param_grid = {'max_depth': [5, 10, 15], 'n_estimators': [10, 15, 20]}, cv = 3)
    rf__model.fit(train_X, train_y)
    predictions = rf__model.predict(test_X)
    
    #printing metrics and writing to file
    print('Random Forest')
    print("\t Best max_depth: %d"%rf__model.best_params_['max_depth'])
    print("\t Best n_estimators: %d"%rf__model.best_params_['n_estimators'])
    print_metrics(test_y, predictions, 'Random Forest', name)

    #Gradient Boosting Regression
    gb_model = GradientBoostingRegressor(random_state = 7)
    gb_model.fit(train_X, train_y)
    predictions = gb_model.predict(test_X)
    
    #printing metrics and writing to file
    print('Gradient Boosted Trees')
    print_metrics(test_y, predictions, 'Gradient Boosted Trees', name)

def get_metrics_classification(y, yhat, model, name, first = False):
    
    #printing metrics of the model
    print("\t accuracy: %.3f"%accuracy_score(y, yhat))
    print("\t weightedRecall: %.3f"%recall_score(y, yhat, average = 'weighted'))
    print("\t weightedPrecision: %.3f"%precision_score(y, yhat, average = 'weighted'))
    
    row = [model, round(accuracy_score(y, yhat), 2)
                  , round(recall_score(y, yhat, average = 'weighted'), 2)
                   , round(precision_score(y, yhat, average = 'weighted'), 2)]
    
    file_name = name + '_metric.csv'
    
    if first:
        mode = 'w'
    else:
        mode = 'a'
    
    with open(file_name, mode) as file:
        writer = csv.writer(file)
        if first:
            header = ['model', 'accuracy', 'weightedRecall', 'weightedPrecision']
            writer.writerow(header)
        writer.writerow(row)


def fit_classification_models(train_X, train_y, test_X, test_y, name, classes = 2):
    
    #Logistic Regression
    en_model = SGDClassifier(loss = 'log', penalty = 'elasticnet')
    en__model = GridSearchCV(en_model, param_grid = {'alpha': [0.0001, 0.25, 0.5], 'l1_ratio':[0, 0.25, 0.5]})
    en__model.fit(train_X, train_y)
    predictions = en__model.predict(test_X)
    
    #printing metrics and writing to file
    print('Linear Regression')
    print("\t Best alpha: %.2f"%en__model.best_params_['alpha'])
    print("\t Best l1_ratio: %.2f"%en__model.best_params_['l1_ratio'])
    get_metrics_classification(test_y, predictions, 'Linear Regression', name, True)
    


    #Decision Tree Classification
    dtc_model = DecisionTreeClassifier(random_state = 7)
    dtc__model = GridSearchCV(dtc_model, param_grid = {'max_depth': [5, 10, 15]}, cv = 3)
    dtc__model.fit(train_X, train_y)
    prediction = dtc__model.predict(test_X)
    
    #printing metrics and writing to file
    print('Decision Tree')
    print("\t Best max_depth: %d"%dtc__model.best_params_['max_depth'])
    get_metrics_classification(test_y, prediction, 'Decision Tree', name)
    


    #Random Forest Classification
    rfc_model = RandomForestClassifier()
    rfc__model = GridSearchCV(rfc_model, param_grid = {'max_depth': [5, 10, 15], 'n_estimators': [10, 15, 20]}, cv = 3)
    rfc__model.fit(train_X, train_y)
    prediction = rfc__model.predict(test_X)
    
    #printing metrics and writing to file
    print('Random Forest')
    print("\t Best max_depth: %d"%rfc__model.best_params_['max_depth'])
    print("\t Best n_estimators: %d"%rfc__model.best_params_['n_estimators'])
    get_metrics_classification(test_y, prediction, 'Random Forest', name)


    #Naive Bayes Clasiification
    nbc_model = GaussianNB()
    nbc__model = GridSearchCV(nbc_model, param_grid = {'var_smoothing': [.5, 1, 2]}, cv = 3)
    nbc__model.fit(train_X, train_y)
    prediction = nbc__model.predict(test_X)
    #printing metrics and writing to file
    print('Naive Bayes')
    get_metrics_classification(test_y, prediction, 'Naive Bayes', name)

    if classes == 2:
        #Gradient Boosting Classification
        gb_model = GradientBoostingClassifier()
        gb_model.fit(train_X, train_y)
        prediction = gb_model.predict(test_X)
        
        #printing metrics and writing to file
        print('Gradient Boosting')
        get_metrics_classification(test_y, prediction, 'Gradient Boosting', name)
