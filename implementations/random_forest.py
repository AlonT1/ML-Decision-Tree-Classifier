import random

import numpy as np
import pandas as pd

from implementations import decision_tree as dt

pd.set_option("display.width", 800)
pd.set_option("max_columns", 25)
random.seed(0)


def random_forest_generator(train_df, task, max_depth=5, min_samples=2, n_estimators=4, n_subspace_samples=300,
                            max_features=4):
    """n_estimators - number of trees in the forest.
    n_subspace_samples - no. of samples(rows) in the derived sub-data from the main data. including duplicates.
    max_features - no. of features in the sub data
    max_depth - max depth of the tree
    task - classification or regression"""
    forest = []
    for i in range(int(n_estimators)):
        df_bootstrap = bootstrapping(train_df, n_subspace_samples)
        tree = dt.decision_tree_generator(df_bootstrap, max_depth=max_depth, min_samples=min_samples,
                                          random_subspace=max_features, task=task)
        forest.append(tree)
    return forest


def bootstrapping(train_df, n_subspace_samples):
    """Generates and returns a sub dataframe from train_df, including duplicates, with n_subspace_samples (rows)"""
    bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=n_subspace_samples)
    df_bootstrap = train_df.iloc[bootstrap_indices]
    return df_bootstrap


def generate_predictions(df, forest):
    """Every tree in the forest is fed with a df in order to generate predictions for the samples it contains.
    The predictions are stored in a data frame, where each column represents a tree with its predictions.
    Afterwards, we combine the results into a single column by taking the mode of each row (most repeating value)"""
    dic_predictions = {}
    for i in range(len(forest)):
        column_name = "tree_".format(i)
        predictions = dt.generate_predictions(df, tree=forest[i])
        dic_predictions[column_name] = predictions
    df_predictions = pd.DataFrame(dic_predictions)
    forest_predictions = df_predictions.mode(axis=1)[0]
    return forest_predictions
