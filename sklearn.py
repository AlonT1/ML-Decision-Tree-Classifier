"""ML_EX1_204194302_307885921
This module runs SKlearn algorithm"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, mean_squared_error

import numpy as np

pd.set_option("display.width", 800)
pd.set_option("display.max_columns", 20)

# Global variables
dataset_name = "adult_income.xlsx"
train_end = 19035
val_start = 19035
val_end = 24131
test_start = 24131


def data_preparation(df, target_column, include_columns):
    label_encoder = preprocessing.LabelEncoder()
    for i in [1, 2, 4, 5, 6, 7, 8, 12]:
        df_adult_income.iloc[:, i] = label_encoder.fit_transform(df_adult_income.iloc[:, i])

    X_train = df_adult_income.iloc[0:train_end, include_columns]
    y_train = df_adult_income.iloc[0:train_end, target_column]
    X_val = df_adult_income.iloc[val_start:val_end, include_columns]
    y_val = df_adult_income.iloc[val_start:val_end, target_column]
    X_test = df_adult_income.iloc[test_start:, include_columns]
    y_test = df_adult_income.iloc[test_start:, target_column]

    return X_train, y_train, X_val, y_val, X_test, y_test


def task_manager(X_train, y_train, X_val, y_val, X_test, y_test, main_algo, task, param_grid, msg):
    # choosing accuracy score algorithm depending on the task - classification / regression
    accuracy_algo = accuracy_score if task == "classification" else mean_squared_error

    # default settings
    main_algo.fit(X_train, y_train)
    y_pred = main_algo.predict(X_test)
    accuracy = accuracy_algo(y_test, y_pred)
    print(msg + " - default settings:", accuracy)

    # grid_searchcv optimization
    X = pd.concat([X_train, X_val], axis=0)
    y = pd.concat([y_train, y_val], axis=0)
    gs_scoring = "accuracy" if task == "classification" else "neg_mean_squared_error"
    best_estimator = GridSearchCV(main_algo, param_grid, cv=5, scoring=gs_scoring)
    best_estimator.fit(X, y)
    y_pred = best_estimator.predict(X_test)
    accuracy = accuracy_algo(y_test, y_pred)
    print(msg + " - Grid Search:", accuracy)


if __name__ == "__main__":
    df_adult_income = pd.read_excel(dataset_name, sheet_name=0)

    param_grid_tree = {'max_depth': [2, 3, 4, 5, 6, 7],
                       'min_samples_split': [5, 10, 15, 20]}

    param_grid_forest = {
        'n_estimators': [100, 150],  # The number of trees in the forest.
        'max_depth': [None, 50, 60],  # The maximum depth of the tree.
        'max_features': ['sqrt', None],  # the number of features to consider when looking for the best split
        'min_samples_split': [2, 5],  # The minimum number of samples required to split an internal node
        'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees.
    }

    print("\nSklearn Classifications: \n============================")
    # Classification with decision tree - Including non-optimized and grid-search optimized
    columns = list(range(0, 13))
    X_train, y_train, X_val, y_val, X_test, y_test = data_preparation(df_adult_income.copy(), 13, columns)
    task_manager(X_train, y_train, X_val, y_val, X_test, y_test, main_algo=DecisionTreeClassifier(),
                 task="classification", param_grid=param_grid_tree, msg="Decision tree classification")
    # Classification with random forest - Including non-optimized and grid-search optimized
    task_manager(X_train, y_train, X_val, y_val, X_test, y_test, main_algo=RandomForestClassifier(),
                 task="classification", param_grid=param_grid_forest, msg="Random forest classification")

    print("\nSklearn Regressions: \n============================")
    columns = [0, 1, 4, 6, 7, 8, 9, 10, 11, 12]
    X_train, y_train, X_val, y_val, X_test, y_test = data_preparation(df_adult_income.copy(), 3, columns)
    task_manager(X_train, y_train, X_val, y_val, X_test, y_test, main_algo=DecisionTreeRegressor(),
                 task="regression", param_grid=param_grid_tree, msg="Decision tree regression")
    # Classification with random forest - Including non-optimized and grid-search optimized
    task_manager(X_train, y_train, X_val, y_val, X_test, y_test, main_algo=RandomForestRegressor(),
                 task="regression", param_grid=param_grid_forest, msg="Random forest regression")
