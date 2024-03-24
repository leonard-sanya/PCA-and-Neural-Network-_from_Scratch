import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, n_component):
        self.n_component = n_component
        self.X_std = None
        self.eigen_values = None
        self.eigen_vectors = None

    def fit(self, X):
        # Mean
        X_mean = np.mean(X, axis=0)

        # Standard deviation
        std = np.std(X, axis=0)

        # Standardization/Normalization
        self.X_std = (X - X_mean) / std

        # Covariance Matrix
        Cov_mat = np.cov(self.X_std.T)

        # Eigen values and Eigen vectors
        self.eigen_values, self.eigen_vectors = np.linalg.eig(Cov_mat)

    def transform(self, X, y):
        # Projection Matrix
        P = self.eigen_vectors[:, :self.n_component]

        # Projection of the standardized data
        X_proj = self.X_std.dot(P)

        # Visualization
        plt.title("PC1 vs PC2")
        plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y)
        plt.xlabel('PC1')
        plt.ylabel('PC2')

        cum_explained_variance = np.cumsum(self.eigen_values) / np.sum(self.eigen_values)

        plt.figure()
        plt.plot(np.arange(1, self.X_std.shape[1] + 1), cum_explained_variance, '-o')
        plt.xticks(np.arange(1, self.X_std.shape[1] + 1))
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance')
        plt.grid()
        plt.show()
