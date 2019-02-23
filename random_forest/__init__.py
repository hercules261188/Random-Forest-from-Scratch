import numpy as np

class Node:
    def __init__(self, X, y):
        self.left = None
        self.right = None
        self.min_error = np.inf
        self.split_row_idx = None
        self.split_col_idx = None
        self.split(X, y)

    def split(self, X, y):
        if len(X) == 1:
            return self
        for i in range(X.shape[1]):
            X1 = X[:, i]
            order = np.argsort(X1)
            X1 = X1[order]
            y1 = y[order]
            self.calculate_split_idx(y1, i)
        
        order = np.argsort(X[:, self.split_col_idx])
        X1 = X[order]
        y1 = y[order]
        X_left = X1[:self.split_row_idx]
        y_left = y1[:self.split_row_idx]
        self.left = Node(X_left, y_left)
        X_right = X1[self.split_row_idx:]
        y_right = y1[self.split_row_idx:]
        self.right = Node(X_right, y_right)

    def calculate_split_idx(self, y, var_idx):
        for i in range(len(y)):
            y_left = y[:i]
            y_right = y[i:]
            error = ((y_left - y_left.mean()) ** 2).sum() + ((y_right - y_right.mean()) ** 2).sum()
            if error < self.min_error:
                self.min_error = error
                self.split_row_idx = i
                self.split_col_idx = var_idx


class DecisionTree:

    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = Node(X, y)

    def predict(self, X):
        pass
    
    def score(self, X):
        pass

    
                

    
