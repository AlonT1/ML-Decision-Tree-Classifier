from collections import Counter
import pandas as pd
from reference.node import Node
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


class CustomDecisionTree:
    def __init__(self):
        self.root = None

    def __build_tree(self, dataset):
        curr_impurity = self.__gini(dataset)
        max_info_gain = 0
        chosen_attribute, subsets, chosen_attribute_column = None, None, None
        ncol = len(dataset[0]) - 1
        nrows = len(dataset)
        counter = 0
        for column in range(0, ncol):
            for index in range(0, nrows):
                #time1 = time()
                counter += 1
                attribute = dataset[index][column]
                (subset1, subset2) = self.__divide_dataset(dataset, attribute, column)
               # print(time() - time1)
                p = len(subset1) / len(dataset)
                info_gain = curr_impurity - p * self.__gini(subset1) - (1 - p) * self.__gini(subset2)

                if info_gain > max_info_gain and len(subset1) > 0 and len(subset2) > 0:
                    subsets = (subset1, subset2)
                    max_info_gain = info_gain
                    chosen_attribute = attribute
                    chosen_attribute_column = column

        if max_info_gain > 0:
            left_true = self.__build_tree(subsets[0])
            right_false = self.__build_tree(subsets[1])
            return Node(attribute=chosen_attribute, column=chosen_attribute_column, left_true=left_true,
                        right_false=right_false, classification=None)
        elif max_info_gain == 0:
            return Node(classification=dataset[0][-1])

    def classify(self, sample, node=None):
        if node is None:
            node = self.root
        if node.classification is not None:
            return node.classification
        else:
            sample_value = sample[node.column]
            if sample_value >= node.attribute:
                node = node.left_true
            elif sample_value != node.attribute:
                node = node.right_false
        return self.classify(sample, node)

    def score(self, X_test, y_test):
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values.tolist()
        if isinstance(y_test, pd.Series):
            y_test = y_test.values.tolist()
        predictions = [self.classify(row) for row in X_test]
        return len(set(predictions) & set(y_test)) / len(set(predictions) | set(y_test))

    def __gini(self, dataset):
        impurity = 0
        if len(dataset) > 0:
            label_set = [value[-1] for value in dataset]
            label_count = Counter(label_set)
            total_label_count = sum(label_count.values())
            for count in label_count.values():
                p = count / total_label_count
                impurity += p * (1 - p)
        return impurity

    def __divide_dataset(self, dataset, attribute, column):
        set1, set2 = [], []
        for row in dataset:
            if row[column] >= attribute:
                set1.append(row)
            else:
                set2.append(row)
        return set1, set2

    def fit(self, dataset, target_index=None):
        if len(dataset) == 0:
            raise ValueError("dataset is empty")
        if isinstance(dataset, pd.DataFrame):
            dataset = list(dataset.itertuples(index=False, name=None))
        self.root = self.__build_tree(dataset)


X = pd.read_csv("fishiris.csv", delimiter=",", index_col=False, header=None, skipinitialspace=True)
clf = CustomDecisionTree()
clf.fit(X)

label_encoder = preprocessing.LabelEncoder()
X.iloc[:, -1] = label_encoder.fit_transform(X.iloc[:, -1])
X_train, X_test, y_train, y_test = train_test_split(X.iloc[:, 0:4], X.iloc[:, -1], test_size=0.2, random_state=0)
treeD = tree.DecisionTreeClassifier()
treeD.fit(X_train, y_train)
print("iris sklearn score: ", treeD.score(X_test, y_test))
joined_train = X_train.join(y_train)
joined_test = X_test.join(y_test)
clf.fit(joined_test)
print("iris our score:", clf.score(X_test, y_test))

# def fit(X_train, y_train):
#     if len(X_train) != len(y_train):
#         return False
#     add_node(X_train, y_train)
#
#
# List of Tuples
# employees = [('jack', 34, 'Sydney'),
#              ('Riti', 31, 'Delhi'),
#              ('Aadi', 16, 'New York'),
#              ('Mohit', 32, 'Delhi')]
# df = pd.DataFrame(employees, columns=["Name", "Age", "City"])
#
# print(df.values.tolist())
