import numpy as np

from utils.metrics import pairwise_distance
import utils.func as F


class KMeans:
    """ Kmeans Clustering.

    Features:
        - This class supports both hard and soft k-means
        - This class supports batched data, i.e., cluster data in each batch separately and simultaneously.
    """
    def __init__(self, k: int, data_dim: int, max_iters: int = 200, eps: float = 1e-5,
                 soft: bool = False, soft_temparature: float = 1.0, verbose: bool = False):
        """
        Args:
            k: number of cluster centers
            data_dim: number of dimensions
            max_iters: maximum number of iterations
            eps: iteration stops if |cost_new - cost_old| < eps
            soft: soft kmeans
            soft_temparature: temparature in softmax
            verbose: output information during training
        """
        self.k = k
        self.data_dim = data_dim
        self.max_iters = max_iters
        self.eps = eps
        self.soft = soft
        self.soft_temparature = soft_temparature
        self.verbose = verbose

        self.centers = None

    def calc_weights_cost(self, X: np.ndarray):
        dists = pairwise_distance(X, self.centers, dist_metric='L2')        # (N, k) or (B, N, k)
        if self.soft:
            weights = F.softmax(-dists / self.soft_temparature, axis=-1)    # (N, k) or (B, N, k)
        else:
            weights = F.onehot(np.argmin(dists, axis=-1), length=self.k)    # (N, k) or (B, N, k)
        cost = np.sum(np.sum(weights * dists, axis=-1), axis=-1)
        return weights, cost

    def fit(self, X: np.ndarray):
        """
        Args:
            X: (N, d) or (B, N, d)
        """
        assert X.shape[-1] == self.data_dim
        random_indices = np.random.choice(np.arange(X.shape[-2]), size=self.k)      # (k, )
        self.centers = X[..., random_indices, :]                                    # (k, d) or (B, k, d)

        for it in range(self.max_iters):
            weights, cost = self.calc_weights_cost(X)                               # (N, k) or (B, N, k)
            self.centers = (np.sum(np.expand_dims(X, axis=-2) *
                                   np.expand_dims(weights, axis=-1), axis=-3) /
                            np.expand_dims(np.sum(weights, axis=-2), axis=-1))      # (k, d) or (B, k, d)
            newweights, newcost = self.calc_weights_cost(X)                         # (N, k) or (B, N, k)
            if self.verbose:
                print(f'Iteration {it}: cost={np.max(newcost):.6f}')
            if np.max(np.abs(cost - newcost)) < self.eps:
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: (M, d) or (B, M, d)
        Returns:
            If hard k-means, return assigned clusters' labels with shape of (M, ) or (B, M)
            If soft k-means, return weights of data belonging to each cluster with shape of (M, k) or (B, M, k)
        """
        weights, _ = self.calc_weights_cost(X)
        if self.soft:
            return weights
        else:
            return np.argmax(weights, axis=-1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.predict(X)


def _test():
    X = np.concatenate([np.random.multivariate_normal(mean=[-10, 6], cov=[[2, 0], [0, 4]], size=100),
                        np.random.multivariate_normal(mean=[8, -15], cov=[[6, 0], [0, 4]], size=150),
                        np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=50)], axis=0)
    print(f'X.shape={X.shape}')

    kmeans = KMeans(k=3, data_dim=2, verbose=True)
    soft_kmeans = KMeans(k=3, data_dim=2, soft=True, soft_temparature=1., verbose=True)
    labels = kmeans.fit_predict(X)
    weights = soft_kmeans.fit_predict(X)
    print(f'labels.shape={labels.shape}')
    print(f'weights.shape={weights.shape}')

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].scatter(X[labels == 0][:, 0], X[labels == 0][:, 1], c='C0', s=10)
    ax[0].scatter(X[labels == 1][:, 0], X[labels == 1][:, 1], c='C1', s=10)
    ax[0].scatter(X[labels == 2][:, 0], X[labels == 2][:, 1], c='C2', s=10)
    c = np.log(weights)
    ax[1].scatter(X[:, 0], X[:, 1], c=c[:, 0], cmap='Blues', s=10)
    ax[2].scatter(X[:, 0], X[:, 1], c=c[:, 1], cmap='Oranges', s=10)
    ax[3].scatter(X[:, 0], X[:, 1], c=c[:, 2], cmap='Greens', s=10)
    ax[0].axis('scaled')
    ax[1].axis('scaled')
    ax[2].axis('scaled')
    ax[3].axis('scaled')
    plt.show()
    plt.close(fig)


def _test_batch():
    X = np.stack([np.concatenate([np.random.multivariate_normal(mean=[-10, 6], cov=[[2, 0], [0, 4]], size=100),
                                  np.random.multivariate_normal(mean=[8, -15], cov=[[6, 0], [0, 4]], size=150),
                                  np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=50)], axis=0),
                  np.concatenate([np.random.multivariate_normal(mean=[15, -6], cov=[[3, 0], [0, 4]], size=100),
                                  np.random.multivariate_normal(mean=[8, 10], cov=[[4, 0], [0, 1]], size=100),
                                  np.random.multivariate_normal(mean=[-1, 0], cov=[[1, 0], [0, 1]], size=100)], axis=0),
                  np.concatenate([np.random.multivariate_normal(mean=[-5, 6], cov=[[2, 0], [0, 4]], size=250),
                                  np.random.multivariate_normal(mean=[0, -15], cov=[[2, 0], [0, 3]], size=30),
                                  np.random.multivariate_normal(mean=[10, 0], cov=[[1, 0], [0, 1]], size=20)], axis=0)],
                 axis=0)
    print(f'X.shape={X.shape}')

    kmeans = KMeans(k=3, data_dim=2, verbose=True)
    soft_kmeans = KMeans(k=3, data_dim=2, soft=True, soft_temparature=1., verbose=True)
    labels = kmeans.fit_predict(X)
    weights = soft_kmeans.fit_predict(X)
    print(f'labels.shape={labels.shape}')
    print(f'weights.shape={weights.shape}')

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 4, figsize=(12, 8))
    for i in range(3):
        ax[i][0].scatter(X[i][labels[i] == 0][:, 0], X[i][labels[i] == 0][:, 1], c='C0', s=10)
        ax[i][0].scatter(X[i][labels[i] == 1][:, 0], X[i][labels[i] == 1][:, 1], c='C1', s=10)
        ax[i][0].scatter(X[i][labels[i] == 2][:, 0], X[i][labels[i] == 2][:, 1], c='C2', s=10)
        c = np.log(weights[i])
        ax[i][1].scatter(X[i][:, 0], X[i][:, 1], c=c[:, 0], cmap='Blues', s=10)
        ax[i][2].scatter(X[i][:, 0], X[i][:, 1], c=c[:, 1], cmap='Oranges', s=10)
        ax[i][3].scatter(X[i][:, 0], X[i][:, 1], c=c[:, 2], cmap='Greens', s=10)
        ax[i][0].axis('square')
        ax[i][1].axis('square')
        ax[i][2].axis('square')
        ax[i][3].axis('square')
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    _test()
    _test_batch()
