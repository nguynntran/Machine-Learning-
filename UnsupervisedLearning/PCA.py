import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None

    def fit(self, X):
        # Centerlize to the origin by mean
        self.mean = np.mean(X, axis = 0)
        X = X - self.mean

        #covariance
        cov = np.cov(X.T)

        #eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        eigenvectors = eigenvectors.T

        # sort eigenvalues and eigenvectors
        idx = np.argsort(eigenvalues)[::-1] # descending order, arg returns the indices that would sort the array
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[idx]

        self.components = eigenvectors[:self.n_components]

    def transform(self, X):
        #project data
        X = X - self.mean
        return np.dot(X, self.components.T)
    
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    import matplotlib.pyplot as plt

    data = load_iris()
    X = data.data
    y = data.target

    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Iris Dataset')
    plt.show()

    print(X.shape)
    print(X_pca.shape)