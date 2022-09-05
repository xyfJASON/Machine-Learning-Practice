import numpy as np
import sklearn.metrics


def pairwise_distance(X1: np.ndarray, X2: np.ndarray, dist_metric: str) -> np.ndarray:
    """ Calculate pairwise distance.

    Args:
        X1: (..., N, d)
        X2: (..., M, d)
        dist_metric: distance metric, one of {'L1', 'L2', 'Manhattan', 'Euclidean', 'Chebyshev', 'Cosine'}

    Note that cosine distance is defined as 1.0 - cosine similarity

    Returns:
        Pairwise distance, (..., N, M)
    """
    assert dist_metric in ['L1', 'L2', 'Manhattan', 'Euclidean', 'Chebyshev', 'Cosine']
    assert X1.shape[-1] == X2.shape[-1]
    if dist_metric == 'Cosine':
        dists = np.sum(np.expand_dims(X1, axis=-2) * np.expand_dims(X2, axis=-3), axis=-1)
        dists = dists / (np.linalg.norm(X1, axis=-1, keepdims=True) * np.linalg.norm(X2, axis=-1, keepdims=True).transpose((-1, -2)))
        dists = 1. - dists
        return dists
    else:
        if dist_metric in ['L1', 'Manhattan']:
            order = 1
        elif dist_metric in ['L2', 'Euclidean']:
            order = 2
        elif dist_metric == 'Chebyshev':
            order = np.inf
        else:
            raise ValueError
        dists = np.linalg.norm(np.expand_dims(X1, axis=-2) - np.expand_dims(X2, axis=-3), ord=order, axis=-1)
    return dists


def _test():
    X1 = np.random.randn(3, 10)
    X2 = np.random.randn(5, 10)

    from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
    print(np.max(np.abs(pairwise_distance(X1, X2, 'L2') - euclidean_distances(X1, X2))))
    print(np.max(np.abs(pairwise_distance(X1, X2, 'L1') - manhattan_distances(X1, X2))))
    print(np.max(np.abs(pairwise_distance(X1, X2, 'Cosine') - cosine_distances(X1, X2))))


if __name__ == '__main__':
    _test()
