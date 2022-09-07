import numpy as np


class MultivariateNormal:
    """ Multivariate Normal (Gaussian) Distribution. """
    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        """
        Args:
            mean: (d, )
            cov: (d, d)
        """
        self.mean = mean
        self.cov = cov
        self.d = mean.shape[-1]
        assert cov.shape == (self.d, self.d)

    def logpdf(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: (M, d)
        Returns:
            log(pdf(X)), (M, )
        """
        meanX = (X - np.expand_dims(self.mean, -2))    # (M, d)
        return (- 1/2 * self.d * np.log(2 * np.pi)
                - 1/2 * np.linalg.slogdet(self.cov)[1]
                - 1/2 * np.squeeze(np.expand_dims(meanX, -2) @
                                   np.expand_dims(np.linalg.inv(self.cov), -3) @
                                   np.expand_dims(meanX, -1), (-1, -2)))

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: (M, d)
        Returns:
            pdf(X), (M, )
        """
        return np.exp(self.logpdf(X))


MultivariateGaussian = MultivariateNormal


def _test():
    mean = np.random.rand(5)
    X = np.random.rand(10, 5)
    cov = np.transpose(X - mean, (-1, -2)) @ (X - mean)
    X = np.random.rand(10, 5)

    from scipy.stats import multivariate_normal
    guassian = MultivariateGaussian(mean, cov)
    guassian2 = multivariate_normal(mean, cov)
    print(guassian.logpdf(X))
    print(guassian2.logpdf(X))


if __name__ == '__main__':
    _test()
