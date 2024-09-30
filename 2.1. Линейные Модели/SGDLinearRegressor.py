import numpy as np
from sklearn.base import RegressorMixin

class SGDLinearRegressor(RegressorMixin):
    def __init__(self,
                 lr=0.01, regularization=1., delta_converged=1e-3, max_steps=1000,
                 batch_size=64):
        self.lr = lr
        self.regularization = regularization
        self.max_steps = max_steps
        self.delta_converged = delta_converged
        self.batch_size = batch_size

        self.W = None
        self.b = None

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self.b, self.W = 0, np.random.rand(n_features)

        for epoch in range(self.max_steps):
            idxs = np.random.choice(np.arange(n_samples), self.batch_size)
            
            X_batch = X[idxs]
            Y_batch = Y[idxs]

            y_pred = X_batch @ self.W + self.b
            grad_w = 2 * (X_batch.T @ (y_pred - Y_batch) / self.batch_size + self.regularization * self.W)
            grad_b = 2 * np.sum(y_pred - Y_batch) / self.batch_size

            previous_W = self.W.copy()
            self.W -= self.lr * grad_w
            self.b -= self.lr * grad_b
            
            if np.linalg.norm(self.W - previous_W) < self.delta_converged: #НАДО УЧИТЫВАТЬ РАЗНИЦУ БАЙАСА?
                break

        return self
                

    def predict(self, X):
        return X @ self.W + self.b