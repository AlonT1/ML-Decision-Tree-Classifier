from itertools import product

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from implementations import decision_tree as dt
from implementations import random_forest as rf


def rate_model_accuracy(val_df, model, func_prediction, task):
    """This function is used in grid search in order to score a model."""
    y_pred = func_prediction(val_df, model)
    accuracy_algo = accuracy_score if task == "classification" else mean_squared_error
    accuracy = accuracy_algo(y_true=val_df["label"], y_pred=y_pred)
    return accuracy


def grid_search(parameters, train_df, val_df, task, is_tree):
    """" Grid search for optimal parameter search. For each combination of given parameters, the function
    creates a model, trains it  on train_df, and tests it on val_df. afterwards the model receives an accuracy
    score. Finally, based on this score, the best model is chosen and returned back."""
    print("Please wait. Performing grid search to optimize the " + task + " task...", end="")

    grid = {"model_accuracy": []}
    for param in parameters:
        grid[param] = []

    parameters_lists = list(parameters.values())
    for parameters_combo in product(*parameters_lists):
        if is_tree:
            tree = dt.decision_tree_generator(train_df, task, *parameters_combo)
            model_score = rate_model_accuracy(val_df, tree, dt.generate_predictions, task)
        else:  # else tree
            forest = rf.random_forest_generator(train_df, task, *parameters_combo)
            model_score = rate_model_accuracy(val_df, forest, rf.generate_predictions, task)
        # update score grid

        grid["model_accuracy"].append(model_score)
        grid["max_depth"].append(parameters_combo[0])
        grid["min_samples"].append(parameters_combo[1])
        if not is_tree:
            grid["n_estimators"].append(parameters_combo[2])
            grid["n_subspace_samples"].append(parameters_combo[3])
            grid["max_features"].append(parameters_combo[4])
        print(".", end="")

    print()  # print new line for readability
    # The dictionary is transformed into a dataframe, and sorted based on model_accuracy column.
    grid_df = pd.DataFrame(grid, dtype=int)
    grid_df.sort_values("model_accuracy", ascending=False, inplace=True)
    best_params = grid_df.iloc[0, 1:]  # choosing 1'st row - the most accurate model
    # print(grid_df)
    if is_tree:
        tree = dt.decision_tree_generator(train_df, task, *best_params)
        return tree
    else:  # else forest
        forest = rf.random_forest_generator(train_df, task, *best_params)
        return forest
