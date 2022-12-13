import numpy as np

# global parameters
data = np.loadtxt("meteo1.csv").astype(int)
V = 3  # number of observable states
H = 3  # number of Markov models


def condp(data):
    return data / np.sum(data, axis=0)


def condexp(logp):
    return condp(np.exp(logp - np.max(logp, 0)))


def uniform_initialization():
    ph = np.full((H, 1), 1/H)
    pv1gh = np.full((V, H), 1/V)
    pvgvh = np.full((V, V, H), 1/V)

    return ph, pv1gh, pvgvh


def random_initialization():
    uniform = np.random.uniform
    ph = uniform(size=(H, 1))
    pv1gh = uniform(size=(V, H))
    pvgvh = uniform(size=(V, V, H))

    return condp(ph), condp(pv1gh), condp(pvgvh)


def prior_pvgvh():
    pvgvh = np.zeros((V, H))
    for h in range(H):
        r, c = np.where(data[:, :-1] == h)
        n = data[(r, c + 1)]
        nc = np.array([np.count_nonzero(n == i) for i in range(V)])
        pvgvh[:, h] = nc

    pvgvh /= np.sum(pvgvh, axis=0)
    return np.broadcast_to(pvgvh, (V, V, H))


# ph, pv1gh, _ = uniform_initialization()
# pvgvh = prior_pvgvh()
ph, pv1gh, pvgvh = random_initialization()

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

print("")
print("")

phgv = np.array(ph_old)

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

print(f"\nlog liklihood for these parameters: {loglik:.4f}")

print("\nposterior for first 10 rows:")
print(np.around(phgv, 4)[:10])
