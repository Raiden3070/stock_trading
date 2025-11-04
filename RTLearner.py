import numpy as np
import random

class RTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def author(self):
        return 'jkim3070'

    def add_evidence(self, data_x, data_y):
        self.tree = self.build_tree(data_x, data_y)
        if self.verbose:
            print(f"DTLearner size: \n{self.tree.shape}\n DTLearner tree: \n {self.tree}")

    def query(self, values):
        predicted = []
        for value in values:
            predicted.append(self.query_value(value))
        return np.asarray(predicted)

    def query_value(self, value):
        node = 0
        while True:
            if node >= self.tree.shape[0]:
                return 'Node Error'
            tree_pos = self.tree[node]
            if np.isnan(tree_pos[0]):
                return tree_pos[1]
            elif value[int(tree_pos[0])] <= tree_pos[1]:
                node += 1
            else:
                node += int(tree_pos[3])

    def build_tree(self, data_x, data_y):
        def majority_vote(y):
            vals, counts = np.unique(y, return_counts=True)
            return vals[np.argmax(counts)]

        if data_x.shape[0] <= self.leaf_size:
            # Leaf: predict majority class
            return np.asarray([np.nan, majority_vote(data_y), np.nan, np.nan])

        if np.all(np.isclose(data_y, data_y[0])):
            return np.asarray([np.nan, data_y[0], np.nan, np.nan])
        random_index = random.randrange(data_x.shape[1])
        data1, data2 = random.sample(range(data_x.shape[0]), 2)
        splitval = (data_x[data1][random_index] + data_x[data2][random_index]) / 2

        left_data = data_x[:, random_index] <= splitval

        if np.all(np.isclose(left_data, left_data[0])):
            return np.asarray([np.nan, majority_vote(data_y), np.nan, np.nan])

        right_data = np.logical_not(left_data)

        lefttree = self.build_tree(data_x[left_data], data_y[left_data])
        righttree = self.build_tree(data_x[right_data], data_y[right_data])

        if lefttree.ndim == 1:
            root = np.asarray([random_index, splitval, 1, 2])
        else:
            root = np.asarray([random_index, splitval, 1, lefttree.shape[0] + 1])

        return np.vstack((root, lefttree, righttree))
