import numpy as np
from numpy.typing import NDArray

# global parameters
V = 4  # number of observable states
H = 2  # number of Markov models


def condp(data: NDArray) -> NDArray:
    """Return a conditional distribution from a data array

    Note that all columns sum to 1

    Args:
        data: An array of data

    Returns:
        NDArray: the conditional distribution
    """
    return data / np.sum(data, axis=0)


def condexp(logp: NDArray) -> NDArray:
    """Compute the exponential of the log probability

    Args:
        logp: log probability

    Returns:
        NDArray: conditional distribution
    """
    return condp(np.exp(logp - np.max(logp, 0)))


def uniform_initialisation() -> tuple[NDArray, NDArray, NDArray]:
    """Uniformly initialize the parameters

    Returns:
        tuple[NDArray, NDArray, NDArray]: uniform p(h),
            p(v_1|h), p(v_t|v_{t - 1},h)
    """
    ph = np.full((H, 1), 1 / H)
    pv1gh = np.full((V, H), 1 / V)
    pvgvh = np.full((V, V, H), 1 / V)

    return ph, pv1gh, pvgvh


def random_initialisation() -> tuple[NDArray, NDArray, NDArray]:
    """Randomly initialize the parameters

    Taken from a uniform distribution

    Returns:
        tuple[NDArray, NDArray, NDArray]: random p(h),
            p(v_1|h), p(v_t|v_{t-1},h)
    """
    uniform = np.random.uniform
    ph = uniform(size=(H, 1))
    pv1gh = uniform(size=(V, H))
    pvgvh = uniform(size=(V, V, H))

    return condp(ph), condp(pv1gh), condp(pvgvh)


def run_em(
    data: NDArray,
    ph: NDArray,
    pv1gh: NDArray,
    pvgvh: NDArray,
    max_iters: int = 20,
) -> tuple:
    """Run the EM algorithm and return learned params

    Args:
        data: the sequence data
        ph: the initialised p(h)
        pv1gh: the initialised p(v_1|h)
        pvgvh: the initialised p(v_t|v_{t-1}, h)
        max_iters: a maximum number of iterations to run the algorithm

    Returns:
        tuple: a tuple of the following data
            ph - learned p(h)
            pv1gh - learned p(v_1|h)
            pvgvh - learned p(v_t|v_{t-1}, h)
            phgv - the calculated posterior p(h|v)
            llik - the log likelihoods at each step of the algorithm
    """
    max_iters = 20
    llik = []
    for iter in range(max_iters):
        # e-step
        ph_stat = np.zeros((H, 1))
        pv1gh_stat = np.zeros((V, H))
        pvgvh_stat = np.zeros((V, V, H))
        loglik = 0

        ph_old = []
        for n in range(data.shape[0]):
            T = data.shape[1]

            lph_old = np.log(ph) + np.log(pv1gh[data[n][0]])[:, np.newaxis]

            for t in range(1, T):
                lph_old += np.log(pvgvh[data[n, t], data[n, t - 1]])[:, np.newaxis]

            ph_old.append(condexp(lph_old))
            loglik += np.log(np.sum(np.exp(lph_old)))

            ph_stat += ph_old[n]

            pv1gh_stat[data[n][0]] += np.squeeze(ph_old[n])

            for t in range(1, T):
                pvgvh_stat[data[n, t], data[n, t - 1], :] += np.squeeze(ph_old[n])

        llik.append(loglik)

        # m-step
        ph = condp(ph_stat)
        pv1gh = condp(pv1gh_stat)
        pvgvh = condp(pvgvh_stat)

        print(f"end of iteration: {iter}, current loglik: {loglik:.5f}")

    phgv = np.array(ph_old)

    return ph, pv1gh, pvgvh, phgv, llik


ph_0, pv1gh_0, pvgvh_0 = random_initialisation()

data = np.loadtxt("genes_int.txt").astype(int)

ph, pv1gh, pvgvh, phgv, llik = run_em(data=data, ph=ph_0, pv1gh=pv1gh_0, pvgvh=pvgvh_0)

seq_1 = []
seq_2 = []
for idx, row in enumerate(np.squeeze(phgv)):
    if row[0] > 0.5:
        seq_1.append(data[idx])
    else:
        seq_2.append(data[idx])


seq_1_chars = []
for seq in seq_1:
    char_sec = ""
    for el in seq:
        if el == 0:
            char = "A"
        elif el == 1:
            char = "C"
        elif el == 2:
            char = "G"
        else:
            char = "T"
        char_sec += char
    seq_1_chars.append(char_sec)

seq_2_chars = []
for seq in seq_2:
    char_sec = ""
    for el in seq:
        if el == 0:
            char = "A"
        elif el == 1:
            char = "C"
        elif el == 2:
            char = "G"
        else:
            char = "T"
        char_sec += char
    seq_2_chars.append(char_sec)

with open("gene_clustered_4.txt", "w") as f:
    for seq in seq_1_chars:
        f.write(f"{seq}\n")
    f.write("\n\n")
    for seq in seq_2_chars:
        f.write(f"{seq}\n")
