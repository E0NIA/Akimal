import pandas as pd
import numpy as np

data = pd.read_csv('data/animal_small.csv')
print(data)

class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor '''

        # for decision node
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # for leaf node
        self.value = value


class DecisionTreeClassifier():

    def __init__(self, min_samples_split = 1):
        
        # inititalize the root of the tree
        self.root = None

        # stopping conditions
        self.min_samples_split = min_samples_split

    
    def build_tree(self, dataset):

        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)

        # split until stopping conditions are met
        if num_samples >= self.min_samples_split:

            best_split = self.get_best_split(dataset, num_features)

            if best_split['info_gain'] > 0:

                left_subtree = self.build_tree(best_split['dataset_left'])

                right_subtree = self.build_tree(best_split['dataset_right'])

                return Node(best_split['feature'],best_split['threshold'], left_subtree, right_subtree, best_split['info_gain'])

        leaf_value = self.calculate_leaf_value(Y)

        return Node(value=leaf_value)


    def get_best_split(self, dataset, num_features):

        best_split = {}
        max_info_gain = -float('inf')

        for feature in range(num_features):

            for threshold in range(2):

                dataset_left, dataset_right = self.split(dataset, feature, threshold)

                if len(dataset_left) > 0 and len(dataset_right) > 0:

                    y, left_y, right_y = dataset[:,-1], dataset_left[:,-1], dataset_right[:,-1]

                    curr_info_gain = self.information_gain(y,left_y, right_y)

                    if curr_info_gain > max_info_gain:
                        best_split["feature"] = feature
                        best_split['threshold'] = threshold
                        best_split['dataset_left'] = dataset_left
                        best_split['dataset_right'] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        return best_split


    def split(self, dataset, feature, threshold):

        dataset_left = np.array([row for row in dataset if int(row[feature]) <= threshold])
        dataset_right = np.array([row for row in dataset if int(row[feature]) > threshold])
        return dataset_left, dataset_right

    def information_gain(self, parent, l_child, r_child):
        ''' function to compute information gain '''

        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        
        return gain

    def gini_index(self, y):
        ''' function to compute gini index '''

        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini

    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''

        Y = list(Y)
        return min(Y, key=Y.count)

    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''

        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature), "<=",
                  tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

    def fit(self, X, Y):
        ''' function to train the tree '''

        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)


X = data.iloc[:, 1:].values
Y = data.iloc[:, 0].values.reshape(-1, 1)

classifier = DecisionTreeClassifier(min_samples_split=2)
classifier.fit(X, Y)
classifier.print_tree()
