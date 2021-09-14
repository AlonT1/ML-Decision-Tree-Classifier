import random

import numpy as np
import pandas as pd

pd.set_option("display.width", 1000)
pd.set_option("display.max_columns", 30)
random.seed(0)


def arrange_label_column(df, target_column):
    """moves target_column to the end of the data frame, and renames it 'label'"""
    if isinstance(target_column, str):
        df = df.assign(label=df.pop(target_column))
    else:  # if integer index is given
        df = df.assign(label=df.drop(df.columns[target_column]))
    return df


def determine_feature_type(df):
    """ determines if the feature is categorical or continuous. Done by looping each column, checking if samples are
     int or str, and comparing the count of unique values against a threshold"""
    feature_types = []
    n_unique_values_threshold = 15
    for feature in df.columns:  # for each column in dataset
        if feature != "label":  # ensuring not last column
            unique_values = df[feature].unique()
            example_value = unique_values[0]
            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_threshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    return feature_types


def decision_tree_generator(df, task, max_depth=5, min_samples=2, counter=0, random_subspace=None):
    # min_samples - number of rows in incoming dataset.
    # max_depth - max num of recursive iterations
    if counter == 0:  # the user inputs pd, we convert it to numpy
        global column_headers, feature_types
        column_headers = df.columns
        feature_types = determine_feature_type(df)
        data = df.values  # Converting to 2d numpy array due to performance reasons
    else:
        data = df  # already numpy array

    # base case
    if (label_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        # if the data contains a single unique label (purity =1) - than this is our classification!
        # o.w if it contains less than min_samples rows, return most repeating label within this rows.
        # max_depth - if an answer isn't found within max_depth recursion - create a leaf
        leaf = generate_leaf(data, task)
        return leaf
    else:
        counter += 1
        potential_splits = get_potential_splits(data, random_subspace)  # all columns dictionary with unique features
        split_column, split_value = generate_best_split(data, potential_splits, task)  # best splitting value & col
        subset1, subset2 = split_data(data, split_column, split_value)  # split according to <= / ==

        # check for empty data
        if len(subset1) == 0 or len(subset2) == 0:
            leaf = generate_leaf(data, task)
            return leaf

        # now that we have the value which caused the best split & its feature (column) name,
        # we'll create the question:
        feature_name = column_headers[split_column]
        type_of_feature = feature_types[split_column]
        if type_of_feature == "continuous":
            question = f"{feature_name} <= {split_value}"
        else:
            question = f"{feature_name} = {split_value}"

        sub_tree = {question: []}
        true_branch = decision_tree_generator(subset1, task, counter, min_samples, max_depth, random_subspace)
        false_branch = decision_tree_generator(subset2, task, counter, min_samples, max_depth, random_subspace)

        if true_branch == false_branch:
            sub_tree = true_branch
        else:
            sub_tree[question].append(true_branch)
            sub_tree[question].append(false_branch)

        return sub_tree


def label_purity(data):
    """if there is only 1 unique label in label column, return true otherwise false"""
    label_column = data[:, -1]
    unique_labels = np.unique(label_column)  # returns list of unique values
    return True if len(unique_labels) == 1 else False


def generate_leaf(data, task):
    """returns the most repeating label out of all labels in label_column"""
    label_column = data[:, -1]
    if task == "regression":
        leaf = np.mean(label_column)
    else:  # classification - find unique labels and their counts - ex. labels: [male,female], counts: [20,150]
        unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
        index = counts_unique_classes.argmax()
        leaf = unique_classes[index]
    return leaf


def get_potential_splits(data, random_subspace):
    """returns a dictionary of potential splits (unique values from each column).
    example -> {0:[1,2,3,4], 1:[male,female]}
    key: index of column, value: unique samples from the column. random_subspace is for forest implementation -
    it defines the size of the subspace of a tree in the forest, meaning no. of features the tree is built upon"""
    potential_splits = {}
    n_columns = data.shape[1]
    column_indices = list(range(n_columns - 1))  # excluding the last column which is the label
    if random_subspace and random_subspace <= len(column_indices):
        column_indices = random.sample(population=column_indices, k=random_subspace)  # choose k features from the data
    for column_index in column_indices:
        values = data[:, column_index]
        unique_values = np.unique(values)
        potential_splits[column_index] = unique_values
    return potential_splits


def generate_best_split(data, potential_splits, task):
    """ iterates through each value in potential_splits, splitting the data, and calculating impurity for
    the split, and constantly saving the value which caused the most pure split (lowest impurity).
    Returns the value & column of the value which splits the data with the lowest impurity"""
    best_overall_metric = 10000
    first_iteration = True
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            subset1, subset2 = split_data(data, split_column=column_index, split_value=value)  # split <=, >
            if task == "regression":
                current_overall_metric = calculate_overall_metric(subset1, subset2, calculate_mse)
            else:
                current_overall_metric = calculate_overall_metric(subset1, subset2, calculate_gini)
            if first_iteration or current_overall_metric <= best_overall_metric:
                first_iteration = False
                best_overall_metric = current_overall_metric  # MSE / Gini - the lowest threshold wins
                best_split_column = column_index
                best_split_value = value
    return best_split_column, best_split_value


def split_data(data, split_column, split_value):
    """Splits data into 2 subsets based on the split_column & split_value which caused the best overall metric"""
    split_column_values = data[:, split_column]
    type_of_feature = feature_types[split_column]
    # data[condition] causes all the rows which meet the condition to be appended into the subset
    if type_of_feature == "continuous":
        subset1 = data[split_column_values <= split_value]
        subset2 = data[split_column_values > split_value]
    else:
        subset1 = data[split_column_values == split_value]
        subset2 = data[split_column_values != split_value]
    return subset1, subset2


def calculate_overall_metric(subset1, subset2, metric_function):
    """takes metric of subset1 1 and subset2 2 & combines both into a single overall metric score.
     This Helps to determine the best split, Since each split has its own score. the lower, the better"""
    total_samples = len(subset1) + len(subset2)
    p_subset1 = len(subset1) / total_samples
    p_subset2 = len(subset2) / total_samples
    overall_metric = (p_subset1 * metric_function(subset1) + p_subset2 * metric_function(subset2))
    return overall_metric


def calculate_mse(data):
    """Returns mean squared error score (squared diff between distance of sample and mean )"""
    actual_values = data[:, -1]
    if len(actual_values) == 0:
        mse = 0
    else:
        prediction = np.mean(actual_values)  # mean of target column is our prediction value of grouped samples.
        mse = np.mean((actual_values - prediction) ** 2)
    return mse


def calculate_gini(data):
    """helper function for calculating entropy of labels from a set of data"""
    # entropy = sig(p * -log2p) - p = chance of a certain unique label, done for all unique labels, and summed (sig)
    label_column = data[:, -1]
    counts = np.unique(label_column, return_counts=True)[1]  # list containing label counts ex: [23, 24]
    probabilities = counts / counts.sum()  # prob. of choosing a label. ex: [1,2] /[ 3,4] = [0.33, 0.5]
    gini = 1 - sum(probabilities ** 2)
    return gini


def generate_predictions(df, tree):
    """returns array which contains labels which serve as our predictions"""
    predictions = [predict_observation(observation, tree) for _, observation in df.iterrows()]
    return predictions


def predict_observation(sample, tree):
    """returns classification. the tree is made up of a dictionary. the function grabs the
    question(key) of current dictionary - if the answer (value of dict) is a classification, it is returned,
    o.w its a dictionary (another subtree), which we must recurse through."""
    question = list(tree.keys())[0]  # grab question. example: "age <= 22"
    feature_name, comparison_operator, value = question.split()
    if comparison_operator == "<=":
        answer = tree[question][0] if sample[feature_name] <= float(value) else tree[question][1]
    else:  # comparison_opeartor is "==", meaning a string
        answer = tree[question][0] if str(sample[feature_name]) == value else tree[question][1]

    # if the answer is not a dictionary, meaning its classification (value/string) return, else its another subtree
    return answer if not isinstance(answer, dict) else predict_observation(sample, answer)
