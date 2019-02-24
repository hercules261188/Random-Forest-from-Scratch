import numpy as np

class Node:

    def __init__(self, X, y, depth):
        self.left = None
        self.right = None
        self.min_error = np.inf
        self.split_col_idx = None
        self.num_rows = X.shape[0]
        self.num_cols = X.shape[1]
        self.split_val = None
        self.depth = depth + 1
        self.split(X, y)

    def split(self, X, y):
        if len(X) == 1:
            return None
        for i in range(self.num_cols):
            self.calculate_split_idx(X[:, i], y, i)
        
        if self.split_col_idx is None:
            return None
    
        X_left = X[self.filt]
        y_left = y[self.filt]
        self.left = Node(X_left, y_left, self.depth)
        X_right = X[~self.filt]
        y_right = y[~self.filt]
        self.right = Node(X_right, y_right, self.depth)

    def calculate_split_idx(self, X, y, var_idx):
        for x in X:
            filt = X <= x
            y_left = y[filt]
            y_right = y[~filt]
            if len(y_right) == 0:
                continue
            error = ((y_left - y_left.mean()) ** 2).sum() + ((y_right - y_right.mean()) ** 2).sum()
            if error < self.min_error:
                self.min_error = error
                self.split_col_idx = var_idx
                self.split_val = x
                self.filt = filt


class DecisionTree:

    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = Node(X, y, -1)

    def predict(self, X):
        pass
    
    def score(self, X):
        pass
