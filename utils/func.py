import warnings
import numpy as np


def logsumexp(X: np.ndarray, axis: int = 0, keepdims: bool = False):
    m = np.max(X, axis=axis, keepdims=True)
    lse = m + np.log(np.sum(np.exp(X - m), axis=axis, keepdims=True))
    return lse if keepdims else np.squeeze(lse, axis=axis)


def log_softmax(X: np.ndarray, axis: int = 0):
    return X - logsumexp(X, axis=axis, keepdims=True)


def softmax(X: np.ndarray, axis: int = 0):
    return np.exp(log_softmax(X, axis=axis))


def onehot(X: np.ndarray, length: int = 0):
    assert np.min(X) >= 0
    if length < np.max(X) + 1:
        length = np.max(X) + 1
        warnings.warn(f'length is set to X.max()+1={np.max(X)+1}')
    return np.eye(length)[X.reshape(-1)].reshape(*X.shape, length)


def _test():
    X = np.array([[[0, 4, 3], [2, 5, 1]],
                  [[3, 4, 1], [5, 0, 2]]])
    oh = onehot(X, length=10)
    print(oh.shape)
    print(oh)


if __name__ == '__main__':
    _test()
