import numpy as np


class BagLearner(object):

    def __init__(self, learner, bags, kwargs, boost, verbose):

        self.boost = boost
        self.verbose = verbose
        self.learners = []
        for i in range(bags):
            self.learners.append(learner(**kwargs))

    def author(self):
        return 'jkim3070'

    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            bag_index = np.random.choice(range(data_x.shape[0]), data_x.shape[0], replace=True)
            bagged_data_x = data_x[bag_index]
            bagged_data_y = data_y[bag_index]
            learner.add_evidence(bagged_data_x, bagged_data_y)

    def query(self, values):
        # Collect predictions from each learner, shape -> (bags, n_samples)
        predicted = [learner.query(values) for learner in self.learners]
        pred_matrix = np.asarray(predicted)

        # Majority vote per sample without SciPy
        # For each column (sample), pick the value with highest count; tie-break by first encountered
        def majority_vote(col):
            vals, counts = np.unique(col, return_counts=True)
            return vals[np.argmax(counts)]

        majority = np.apply_along_axis(majority_vote, axis=0, arr=pred_matrix)
        return majority

        #return sum(predicted) / len(predicted)
