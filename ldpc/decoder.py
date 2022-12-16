import numpy as np
from numpy.typing import NDArray


def initialise(H: NDArray, y: NDArray, p: float) -> NDArray:
    """Initialise messages from bit to check nodes

    The log-likelihood of the bit node v conditioned on its observed value

    Args:
        H: a parity check matrix
        y: the word to decode
        p: the noise ratio

    Returns:
        NDArray: a matrix of initialized messages from
            bits to checks
    """
    log_lik = np.log((1 - p) / p)
    y_lik = np.where(y == 0, log_lik, -log_lik)
    return H * y_lik


def message_to_check(m_v: NDArray, m_cv: NDArray) -> NDArray:
    """Pass messages from bit nodes to check nodes

    Args:
        m_v: the initial messages from bit to check
        m_cv: the current messages from check to bit

    Returns:
        NDArray: a matrix of messages from bit to check nodes
    """
    # initialise the return matrix as the initial bit to check messages
    m_vc = m_v.copy()

    # iterate through columns (bit nodes)
    for idx, row in enumerate(m_cv.T):
        # find the nonzero elements (the check nodes incident to this bit node)
        non_zeros = np.nonzero(row)
        # calculate the updated message
        row_sums = np.sum(row[non_zeros]) - row[non_zeros]
        # add to the initial messages for this bit node
        m_vc.T[idx][non_zeros] += row_sums

    return m_vc


def check_to_message(m_vc: NDArray) -> NDArray:
    """Pass messages from check nodes to bit nodes

    Args:
        m_vc: the current messages from bit to check

    Returns:
        NDArray: a matrix of messages from check to bit nodes
    """
    m_cv = np.zeros(m_vc.shape)

    # iterate through the rows (check nodes)
    for idx, row in enumerate(m_vc):
        # find the nonzero elements (the bit nodes incident to this check node)
        non_zeros = np.nonzero(row)
        # calculate the updated message
        row_tanh = np.prod(np.tanh(row[non_zeros] / 2))
        prod_tanh = row_tanh / np.tanh(row[non_zeros] / 2)
        # update the row in the return matrix with the updated messages
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
    algorithm, as per LDPC Codes: An Introduction (Shokrollahi) and Information
    Theory, Inference, and Learning Algorithms (MacKay)

    Args:
        H: a parity check matrix
        y: the word to decode
        p: the noise ratio
        max_iters: a maximum number of iterations to reach convergence

    Returns:
        tuple[NDArray, int]: a tuple of the decoded message and a return code
            denoting whether the algorithm converged to an answer.
    """
    return_code = -1
    decoded = y

    # initialise the messages (the first round of messages from bit to check nodes)
    m_v = initialise(H, y, p)
    # do the first round of message passing from check to bit nodes
    m_cv = check_to_message(m_v)

    # iterate through the algorithm
    for _ in range(max_iters):
        # calculate next round of messages from bit to check nodes
        m_vc = message_to_check(m_v, m_cv)

        # generate a tentative decoding
        liks = np.sum(m_vc, axis=0)
        decoded = np.where(liks < 0, 1, 0)
        # see if decoded message is a valid codeword
        if not np.any(H @ decoded % 2):
            # decoded is a valid codeword, we can halt the algorithm
            print("found decoded word")
            return_code = 0
            break

        # decoded is not a valid codeword so we move onto the next round
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

empirical_noise_ratio = 1 - np.count_nonzero(y == decoded) / len(decoded)
print(f"empirical_noise_ratio: {empirical_noise_ratio:.2f}")
