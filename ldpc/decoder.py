import numpy as np
from numpy.typing import NDArray


def initialise_probs(H: NDArray, y: NDArray, p: float) -> NDArray:
    """Initialise probabilities

    The log-likelihood of bit node v conditioned on its observed value.
    These act as the first messages to be passed from bit to check node.

    Args:
        H: a parity check matrix
        y: the word to decode
        p: the noise ratio

    Returns:
        NDArray: a matrix of initialised probabilities
    """
    log_lik = np.log((1 - p) / p)
    y_lik = np.where(y == 0, log_lik, -log_lik)
    return H * y_lik


def bit_to_check(m_v: NDArray, p_mat: NDArray) -> NDArray:
    """Pass messages from bit nodes to check nodes

    Args:
        m_v: the initial messages from bit to check
        p_mat: a matrix of the current probabilities

    Returns:
        NDArray: an updated matrix of probabilities
    """
    # initialise the return matrix as the initial bit to check messages
    new_p_mat = m_v.copy()

    # iterate through columns (bit nodes)
    for idx, col in enumerate(p_mat.T):
        # find the nonzero elements (the check nodes incident to this bit node)
        non_zeros = np.nonzero(col)
        # calculate the updated probabilities
        col_sums = np.sum(col[non_zeros]) - col[non_zeros]
        # add to the initial messages for this bit node
        new_p_mat.T[idx][non_zeros] += col_sums

    return new_p_mat


def check_to_bit(p_mat: NDArray) -> NDArray:
    """Pass messages from check nodes to bit nodes

    Args:
        p_mat: a matrix of the current probabilities

    Returns:
        NDArray: an updated matrix of probabilities
    """
    new_p_mat = np.zeros_like(p_mat)

    # iterate through the rows (check nodes)
    for idx, row in enumerate(p_mat):
        # find the nonzero elements (the bit nodes incident to this check node)
        non_zeros = np.nonzero(row)
        # calculate the updated message
        row_tanh = np.prod(np.tanh(row[non_zeros] / 2))
        prod_tanh = row_tanh / np.tanh(row[non_zeros] / 2)
        # update the row in the return matrix with the updated probabilities
        new_p_mat[idx][non_zeros] = np.log((1 + prod_tanh) / (1 - prod_tanh))

    return new_p_mat


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
    # start off with decoded as the same as the received word
    decoded = y

    # initialise the probabilities matrix to be updated by the algorithm
    p_mat = initialise_probs(H, y, p)
    # freeze the initialised probabilities to be used as the first round of messages
    # from bit to check nodes
    m_v = p_mat.copy()

    # iterate through the algorithm
    for _ in range(max_iters):
        # next round of message passing from check to bit nodes
        p_mat = check_to_bit(p_mat)

        # generate a tentative decoding
        decoded = np.where(np.sum(p_mat, axis=0) < 0, 1, 0)
        # see if decoded is a valid codeword
        if not np.any(H @ decoded % 2):
            # decoded is a valid codeword, we can halt the algorithm
            return_code = 0
            break

        # we don't have a valid codeword, proceed to next round of messages from bit to
        # check nodes
        p_mat = bit_to_check(m_v, p_mat)

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

decoded, success = decode(H, y, 0.1, max_iters=200)

if success == 0:
    print("---- SUCCESSFUL DECODING -----")
    empirical_noise_ratio = 1 - np.count_nonzero(y == decoded) / len(decoded)
    print(f"empirical noise ratio: {empirical_noise_ratio:.2f}")

    original_msg = bytearray(np.packbits(decoded[:248])).decode().strip("\x00")
    print(f"decoded message: {original_msg}")
else:
    print("---- unsuccessful decoding -----")
