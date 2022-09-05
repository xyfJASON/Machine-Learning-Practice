import numpy as np

from utils.metrics import pairwise_distance


class KNN:
    """ K-Nearest-Neighbors Classifier.

    Feature:
        - This class supports batched data, i.e., learning multiple KNNs simultaneously.
    """
    def __init__(self, k: int, dist_metric: str = 'L2'):
        """
        Args:
            k: number of neighbors
            dist_metric: distance metric, one of {'L1', 'L2', 'Manhattan', 'Euclidean', 'Chebyshev'}
        """
        self.k = k
        assert dist_metric in ['L1', 'Manhattan', 'L2', 'Euclidean', 'Chebyshev']
        self.dist_metric = dist_metric
        self.X, self.y = None, None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """ KNN doesn't have an explicit learning process, all it needs to do is to store the training data.

        Args:
            X: (N, d) or (B, N, d)
            y: (N, ) or (B, N, )
        """
        self.X, self.y = X, y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: (M, d) or (B, M, d)
        Returns:
            Predicted labels, (M, ) or (B, M)
        """
        dists = pairwise_distance(X, self.X, self.dist_metric)                          # (M, N) or (B, M, N)
        sortids = np.argsort(dists, axis=-1)                                            # (M, N) or (B, M, N)
        y = np.expand_dims(self.y, -2)                                                  # (1, N) or (B, 1, N)
        y = np.take_along_axis(y, sortids, axis=-1)                                     # (M, N) or (B, M, N)
        y = y[..., :min(self.k, dists.shape[1])]                                        # (M, K) or (B, M, K)
        # find the modes, scipy.stats.mode also does the work
        y = np.apply_along_axis(lambda arr: np.bincount(arr).argmax(), axis=-1, arr=y)  # (M, ) or (B, M)
        return y


def main():
    X = np.array([[2, 3],
                  [5, 4],
                  [9, 6],
                  [4, 7],
                  [8, 1],
                  [7, 2]])
    y = np.array([0, 0, 1, 0, 1, 1])
    k = 1
    dist_metric = 'L2'
    knn = KNN(k=k, dist_metric=dist_metric)
    knn.fit(X, y)
    grids = np.meshgrid(np.linspace(0, 10, 300), np.linspace(0, 10, 300), indexing='xy')
    z = knn.predict(np.stack(grids, axis=2).reshape(-1, 2)).reshape(300, 300)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    ax.contourf(grids[0], grids[1], z, alpha=0.3)
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue')
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='green')
    ax.axis('scaled')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title(f'k={k}, dist_metric={dist_metric}')
    ax.plot()
    plt.show()


if __name__ == '__main__':
    main()
