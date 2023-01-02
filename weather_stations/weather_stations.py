import numpy as np
from numpy.typing import NDArray

# global parameters
V = 3  # number of observable states
H = 3  # number of Markov models


def condp(data: NDArray) -> NDArray:
    """Return a conditional distribution from a data array

    All columns sum to 1

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


def prior_pvgvh(data: NDArray) -> NDArray:
    """Set p(v_t|v_{t-1},h) from the data

    Computes the empirical distribution from the data

    Returns:
        NDArray: a (V,V,H) matrix for the conditional distribution
            of p(v_t|v_{t-1},h) based on the data
    """
    pvgvh = np.zeros((V, H))
    for h in range(H):
        r, c = np.where(data[:, :-1] == h)
        n = data[(r, c + 1)]
        nc = np.array([np.count_nonzero(n == i) for i in range(V)])
        pvgvh[:, h] = nc

    pvgvh /= np.sum(pvgvh, axis=0)
    return np.broadcast_to(pvgvh, (V, V, H))


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

            lph_old = np.log(ph) + np.log(pv1gh[data[n, 0]])[:, np.newaxis]

            for t in range(1, T):
                lph_old += np.log(pvgvh[data[n, t], data[n, t - 1]])[:, np.newaxis]

            ph_old.append(condexp(lph_old))
            loglik += np.log(np.sum(np.exp(lph_old)))

            ph_stat += ph_old[n]

            pv1gh_stat[data[n, 0]] += np.squeeze(ph_old[n])

            for t in range(1, T):
                pvgvh_stat[data[n, t], data[n, t - 1]] += np.squeeze(ph_old[n])

        llik.append(loglik)

        # m-step
        ph = condp(ph_stat)
        pv1gh = condp(pv1gh_stat)
        pvgvh = condp(pvgvh_stat)

        print(f"end of iteration: {iter}, current loglik: {loglik:.5f}")

    phgv = np.array(ph_old)

    return ph, pv1gh, pvgvh, phgv, llik


data = np.loadtxt("meteo1.csv").astype(int)

# uniform initialisation
# ph_0, pv1gh_0, pvgvh_0 = uniform_initialisation()

# uniform_initialisation with prior generated from the data
# ph_0, pv1gh_0, _ = uniform_initialisation()
# pvgvh_0 = prior_pvgvh(data)

# random initialisation
ph_0, pv1gh_0, pvgvh_0 = random_initialisation()

# random_initialisation with prior generated from the data
# ph_0, pv1gh_0, _ = random_initialisation()
# pvgvh_0 = prior_pvgvh(data)

ph, pv1gh, pvgvh, phgv, llik = run_em(
    data=data,
    ph=ph_0,
    pv1gh=pv1gh_0,
    pvgvh=pvgvh_0,
)

print("")
print("")

print("----- learned parameters -----")

print("\np(h):")
print(np.around(ph, 4))

print("\np(v_1|h):")
print(np.around(pv1gh, 4))

pvgvh = np.around(pvgvh, 4)
print("\np(v_t|v_{t-1},h):")
print("h = 1:")
print(pvgvh[:, :, 0])
print("h = 2:")
print(pvgvh[:, :, 1])
print("h = 3:")
print(pvgvh[:, :, 2])

print(f"\nlog likelihood for these parameters: {llik[-1]:.4f}")

print("\nposterior for first 10 rows:")
print(np.around(phgv, 4)[:10])
