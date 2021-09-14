"""ML_EX1_204194302_307885921
This Module runs our implementation of Decision Tree, Regression Tree, and Random Forest"""

import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error

from implementations import decision_tree as dt
from implementations import random_forest as rf
from implementations.grid_searchcv import grid_search

pd.set_option("display.width", 800)
pd.set_option("display.max_columns", 20)

# Global variables
dataset_name = "adult_income.xlsx"
train_end = 19035  # 19035
val_start = 19035  # 19035
val_end = 24131  # 24131
test_start = 24131  # 24131


def data_preparation(df, target_column, drop_cols=None):
    if drop_cols:
        df = df.drop(df.columns[drop_cols], axis=1)
    df = dt.arrange_label_column(df, target_column)

    train_df = df.iloc[0:train_end]
    val_df = df.iloc[val_start:val_end]
    test_df = df.iloc[test_start:]
    return train_df, val_df, test_df


def task_manager(test_df, train_df, val_df, main_algo, task, param_grid, msg):
    if main_algo.__name__ == "decision_tree_generator":  # if tree algorithm
        func_prediction = dt.generate_predictions
        is_tree = True
    else:  # else forest algorithm
        func_prediction = rf.generate_predictions
        is_tree = False

    # choosing accuracy score algorithm depending on the task - classification / regression
    accuracy_algo = accuracy_score if task == "classification" else mean_squared_error

    # default settings
    estimator = main_algo(train_df, task=task)
    y_pred = func_prediction(test_df, estimator)
    accuracy = accuracy_algo(y_true=test_df["label"], y_pred=y_pred)
    print(msg + " - default settings: ", accuracy)

    # grid_search optimization
    best_estimator = grid_search(param_grid, train_df, val_df, task=task, is_tree=is_tree)
    y_pred = func_prediction(test_df, best_estimator)
    accuracy = accuracy_algo(y_true=test_df["label"], y_pred=y_pred)
    print(msg + " - Grid Search: ", accuracy)


if __name__ == "__main__":
    df_adult_income = pd.read_excel(dataset_name, sheet_name=0)

    param_grid_tree = {'max_depth': [2, 4, 6], 'min_samples': [5, 10, 15]}
    param_grid_forest = {'max_depth': [2, 4, 6], 'min_samples': [5, 10, 15], "n_estimators": [4, 8],
                         "n_subspace_samples": [train_end], "max_features": [3]}

    print("\nClassifications with our implementations: \n============================")
    train_df, val_df, test_df = data_preparation(df_adult_income.copy(), ">50K", drop_cols=None)
    # Classification with decision tree - Including non-optimized and grid-search optimized
    task_manager(test_df, train_df, val_df, main_algo=dt.decision_tree_generator,
                 task="classification", param_grid=param_grid_tree, msg="Decision tree classification")
    # Classification with random forest - Including non-optimized and grid-search optimized
    task_manager(test_df, train_df, val_df, main_algo=rf.random_forest_generator,
                 task="classification", param_grid=param_grid_forest, msg="Random forest classification")

    print("\n\nRegressions with our implementations: \n============================")
    train_df, val_df, test_df = data_preparation(df_adult_income.copy(), "education-num", drop_cols=[2, 5, 13])
    # Regression with decision tree - Including non-optimized and grid-search optimized
    task_manager(test_df, train_df, val_df, main_algo=dt.decision_tree_generator,
                 task="regression", param_grid=param_grid_tree, msg="Decision tree regression")
    # Regression with random forest - Including non-optimized and grid-search optimized
    task_manager(test_df, train_df, val_df, main_algo=rf.random_forest_generator,
                 task="regression", param_grid=param_grid_forest, msg="Random forest regression")

    print("\n\nMulticlassifications with our implementations: \n============================")
    train_df, val_df, test_df = data_preparation(df_adult_income.copy(), "education-num", drop_cols=[2, 5, 13])
    # Multiclassification with decision tree - Including non-optimized and grid-search optimized
    task_manager(test_df, train_df, val_df, main_algo=dt.decision_tree_generator,
                 task="classification", param_grid=param_grid_tree, msg="Decision tree Multiclassification")
    # Multiclassification with random forest - Including non-optimized and grid-search optimized
    task_manager(test_df, train_df, val_df, main_algo=rf.random_forest_generator,
                 task="classification", param_grid=param_grid_forest, msg="Random forest Multiclassification")
