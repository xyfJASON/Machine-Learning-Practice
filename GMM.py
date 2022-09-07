import numpy as np

import utils.func as F


class GMM:
    """ Gaussian Mixture Model.

    Feature:
        - This class supports batched data, i.e., build the model in each batch separately and simultaneously.
    """
    def __init__(self, k: int, max_iters: int = 200, eps: float = 1e-5, verbose: bool = False):
        """
        Args:
            k: number of cluster centers
            max_iters: maximum number of iterations
            eps: iteration stops if |Q_new - Q_old| < eps
            verbose: output information during training
        """
        self.k = k
        self.max_iters = max_iters
        self.eps = eps
        self.verbose = verbose

        self.alphas = None
        self.means = None
        self.covs = None

    def calc_weights_Q(self, X: np.ndarray):
        d = X.shape[-1]
        meanX = (np.expand_dims(X, -2) - np.expand_dims(self.means, -3))                        # (N, k, d) or (B, N, k, d)
        log_gauss_prob = (- 1/2 * d * np.log(2 * np.pi)
                          - 1/2 * np.expand_dims(np.linalg.slogdet(self.covs)[1], -2)
                          - 1/2 * np.squeeze(np.expand_dims(meanX, -2) @
                                             np.expand_dims(np.linalg.inv(self.covs), -4) @
                                             np.expand_dims(meanX, -1), (-1, -2)))              # (N, k) or (B, N, k)
        p = F.softmax(np.log(np.expand_dims(self.alphas, -2)) + log_gauss_prob, axis=-1)        # (N, k) or (B, N, k)
        Q = np.sum(p * (np.log(np.expand_dims(self.alphas, -2)) + log_gauss_prob), axis=(-1, -2))
        return p, Q

    def fit(self, X: np.ndarray):
        """
        Args:
            X: (N, d) or (B, N, d)
        """
        d = X.shape[-1]
        # random initialization
        self.alphas = np.random.rand(self.k)                                                # (k, )
        self.alphas /= np.sum(self.alphas)                                                  # (k, )
        random_indices = np.random.choice(np.arange(X.shape[-2]), size=self.k)              # (k, )
        self.means = X[..., random_indices, :]                                              # (k, d) or (B, k, d)
        self.covs = np.tile(np.diag(np.random.rand(d)), (self.k, 1, 1))                     # (k, d, d)
        if X.ndim == 3:  # batched data
            B = X.shape[0]
            self.alphas = np.tile(self.alphas, (B, 1))                                      # (B, k)
            self.covs = np.tile(self.covs, (B, 1, 1, 1))                                    # (B, k, d, d)

        for it in range(self.max_iters):
            p, Q = self.calc_weights_Q(X)                                                   # (N, k) or (B, N, k)
            self.alphas = np.sum(p, axis=-2) / p.shape[-2]                                  # (k, ) or (B, k)
            self.means = (np.sum(np.expand_dims(p, -1) * np.expand_dims(X, -2), axis=-3) /
                          np.expand_dims(np.sum(p, axis=-2), -1))                           # (k, d) or (B, k, d))
            meanX = (np.expand_dims(X, -2) - np.expand_dims(self.means, -3))                # (N, k, d) or (B, N, k, d)
            self.covs = (np.sum(np.expand_dims(p, (-1, -2)) *
                                (np.expand_dims(meanX, -1) @
                                 np.expand_dims(meanX, -2)), axis=-4) /
                         np.expand_dims(np.sum(p, axis=-2), (-1, -2)))                      # (k, d, d) or (B, k, d, d)
            newweights, newQ = self.calc_weights_Q(X)                                       # (N, k) or (B, N, k)
            if self.verbose:
                print(f'Iteration {it}: Q={np.min(newQ):.6f}')
            if np.max(np.abs(Q - newQ)) < self.eps:
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: (M, d) or (B, M, d)
        Returns:
            Return weights of data belonging to each cluster with shape of (M, k) or (B, M, k)
        """
        p, _ = self.calc_weights_Q(X)
        return p

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.predict(X)


def _test():
    X = np.concatenate([np.random.multivariate_normal(mean=[-10, 6], cov=[[2, 1], [1, 14]], size=100),
                        np.random.multivariate_normal(mean=[8, -15], cov=[[6, 4], [4, 4]], size=150),
                        np.random.multivariate_normal(mean=[0, -4], cov=[[2, -1], [-1, 2]], size=50)], axis=0)
    print(f'X.shape={X.shape}')

    gmm = GMM(k=3, verbose=True)
    weights = gmm.fit_predict(X)
    print(f'weights.shape={weights.shape}')

    from utils.stats import MultivariateGaussian
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    print(f'alphas: {gmm.alphas}')

    cmaps = ['Blues', 'Oranges', 'Greens']
    grids = np.meshgrid(np.linspace(np.min(X), np.max(X), 300), np.linspace(np.min(X), np.max(X), 300), indexing='xy')
    for i, (mean, cov) in enumerate(zip(gmm.means, gmm.covs)):
        dist = MultivariateGaussian(mean, cov)
        z = dist.pdf(np.stack(grids, axis=2).reshape(-1, 2)).reshape(300, 300)
        ax[0].contour(grids[0], grids[1], z, cmap=cmaps[i])
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
    X = np.stack([np.concatenate([np.random.multivariate_normal(mean=[-10, 6], cov=[[2, 1], [1, 4]], size=100),
                                  np.random.multivariate_normal(mean=[8, -15], cov=[[6, 4], [4, 4]], size=150),
                                  np.random.multivariate_normal(mean=[0, 0], cov=[[2, -1], [-1, 2]], size=50)], axis=0),
                  np.concatenate([np.random.multivariate_normal(mean=[15, -6], cov=[[3, 0], [0, 4]], size=100),
                                  np.random.multivariate_normal(mean=[8, 10], cov=[[4, 0], [0, 1]], size=100),
                                  np.random.multivariate_normal(mean=[-1, 0], cov=[[1, 0], [0, 1]], size=100)], axis=0),
                  np.concatenate([np.random.multivariate_normal(mean=[-5, 6], cov=[[2, -1], [-1, 4]], size=250),
                                  np.random.multivariate_normal(mean=[0, -15], cov=[[2, 1], [1, 3]], size=30),
                                  np.random.multivariate_normal(mean=[10, 0], cov=[[1, 0], [0, 1]], size=20)], axis=0)],
                 axis=0)
    print(f'X.shape={X.shape}')

    gmm = GMM(k=3, verbose=True)
    weights = gmm.fit_predict(X)
    print(f'weights.shape={weights.shape}')

    from utils.stats import MultivariateGaussian
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 4, figsize=(12, 8))
    for i in range(3):
        cmaps = ['Blues', 'Oranges', 'Greens']
        grids = np.meshgrid(np.linspace(np.min(X), np.max(X), 300), np.linspace(np.min(X), np.max(X), 300), indexing='xy')
        for j, (mean, cov) in enumerate(zip(gmm.means[i], gmm.covs[i])):
            dist = MultivariateGaussian(mean, cov)
            z = dist.pdf(np.stack(grids, axis=2).reshape(-1, 2)).reshape(300, 300)
            ax[i][0].contour(grids[0], grids[1], z, cmap=cmaps[j])
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
    # _test_batch()
