import numpy as np
from numpy.typing import NDArray


def initialise(H: NDArray, y: NDArray, p: float) -> NDArray:
    """Initialise the messages

    Args:
        H: a parity check matrix
        y: the word to decode
        p: the noise ratio

    Returns:
        NDArray: a matrix of initialized q variables
    """
    log_lik = np.log((1 - p) / p)
    y_lik = np.where(y == 0, log_lik, -log_lik)
    return H * y_lik


def message_to_check(m_v: NDArray, m_cv: NDArray) -> NDArray:
    m_vc = m_v.copy()

    for idx, row in enumerate(m_cv.T):
        non_zeros = np.nonzero(row)
        row_sums = np.sum(row[non_zeros]) - row[non_zeros]
        m_vc.T[idx][non_zeros] += row_sums

    return m_vc


def check_to_message(m_v: NDArray) -> NDArray:
    m_cv = np.zeros(m_v.shape)

    for idx, row in enumerate(m_v):
        non_zeros = np.nonzero(row)
        row_tanh = np.prod(np.tanh(row[non_zeros] / 2))
        prod_tanh = row_tanh / np.tanh(row[non_zeros] / 2)
        m_cv[idx][non_zeros] = np.log((1 + prod_tanh) / (1 - prod_tanh))

    return m_cv


def decode(
    H: NDArray,
    y: NDArray,
    p: float,
    max_iters: int = 20,
) -> tuple[NDArray, int]:
    """Decode a word using a parity check matrix

    Implements the Loopy Belief Propagation for Binary Symmetric Channel
    algorithm

    Args:
        H: a parity check matrix
        y: the word to decode
        p: the noise ratio
        max_iters: a maximum number of iterations to reach convergence

    Returns:
        int: [TODO:description]
    """
    return_code = -1
    decoded = y
    m_v = initialise(H, y, p)
    m_cv = check_to_message(m_v)

    for i in range(max_iters):
        m_vc = message_to_check(m_v, m_cv)

        liks = np.sum(m_vc, axis=0)
        decoded = np.where(liks < 0, 1, 0)
        if not np.any(H @ decoded % 2):
            print("found decoded word")
            return_code = 0
            break

        m_cv = check_to_message(m_vc)

    return decoded, return_code


H = np.loadtxt("H1.txt")
y = np.loadtxt("y1.txt")

# H = np.array(
#     [
#         [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
#         [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
#         [0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
#         [0, 0, 0, 1, 0, 0, 1, 0, 1, 1],
#     ]
# )
# y = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 1])

decoded, _ = decode(H, y, 0.1, max_iters=200)
breakpoint()
