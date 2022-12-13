import numpy as np


def echelon(m: np.array) -> np.array:
    """Return the echelon version of a matrix

    Args:
        m: the matrix to be echeloned

    Returns:
        np.array: an echelon version of the given matrix
    """
    if 0 in m.shape:
        # one of the rows or cols is zero, no more actions to take
        return m

    # check if first column is all zero
    if not np.any(m[:, 0]):
        # apply echelon on matrix minus first col
        m_sub = echelon(m[:, 1:])
        # append first col
        return np.c_[m[:, 0], m_sub]

    # find nonzero element indices in first col
    nonzero_idx = np.where(m[:, 0] != 0)[0]

    # if first nonzero element is not the first element,
    # i.e. first element is 0
    first_nonzero = nonzero_idx[0]
    if first_nonzero != 0:
        # add first nonzero row, making the first element 1
        m[0] = (m[0] + m[first_nonzero]) % 2

    # add the first row to the rows with nonzero elements
    # to make them 0
    remaining_nonzero_idx = nonzero_idx[nonzero_idx != 0]
    m[remaining_nonzero_idx] = (m[remaining_nonzero_idx] + m[0]) % 2

    # run echelon on the matrix minus first row
    m_sub = echelon(m[1:])

    # add the first row back and return
    return np.vstack((m[0], m_sub))


def rref(m: np.array) -> np.array:
    """Return the rref of a given matrix

    Args:
        m: the matrix to be rref

    Returns:
        np.array: the rref version of m
    """
    m_ech = echelon(m)

    for i in reversed(range(m_ech.shape[0])):
        row = m_ech[i]

        # if row is all 0s, ignore
        if not np.any(row):
            continue

        # find the leading one col
        leading_one_col = np.argmax(row)

        # find indices of ones above the leading one and add leading one row
        ones = np.where(m_ech[:i, leading_one_col])[0]
        m_ech[ones] = (m_ech[ones] + m_ech[i]) % 2

    return m_ech


def create_encoder_matrix(h: np.array) -> tuple[np.array, np.array]:
    """Generate an encoder matrix for a parity check matrix

    Args:
        h: a parity check matrix

    Returns:
        tuple[np.array, np.array]: a tuple of the rref equivalent
            form of h, and the generator matrix
    """
    H_rref = rref(h)

    m, n = H_rref.shape
    k = n - m

    P = H_rref[:, m:]
    H_hat = np.c_[P, np.identity(m)].astype(int)
    G = np.r_[np.identity(k), P].astype(int)

    return H_hat, G


H = np.array([
    [1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 0],
])


H_hat, G = create_encoder_matrix(H)

# generate some random ts and assert H_hat @ G @ t = 0
ts = np.random.randint(0, 2, size=9).reshape((3, 3))
for t in ts:
    assert not np.any((H_hat @ G @ t) % 2)
