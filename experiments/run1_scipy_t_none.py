from scipy.interpolate import make_smoothing_spline  # Works on the userknots branch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Select a bunch of lambdas
lam = np.logspace(-6, 3, 100)

rng = np.random.default_rng(42)
x = np.sort(rng.uniform(0, 4, 100))
signal = np.sin(x)
white_noise = 0.3 * rng.normal(size=100)  # N(0, sigma ** 2) => sigma = 0.3
y = signal + white_noise

n = len(x)

# Now we have the x & y dataset with n = 100

# Uncomment if dataset not saved.
df = pd.DataFrame({"x": x, "y": y})
df.to_csv("experiment_dataset.csv", index=False)
df = pd.read_csv("experiment_dataset.csv")
x, y = df["x"].to_numpy(), df["y"].to_numpy()

y_hats = {}
identity = np.eye(n)
A = np.zeros((n, n))

v_gcv_lam = []
v_loo_lam = []
trA_all = []

for l in lam:
    spl = make_smoothing_spline(x, y, lam=l)
    y_hat = spl(x)
    y_hats[f"{l}"] = np.asarray(y_hat, dtype=np.float64)

    for i, ek in enumerate(identity):
        ek = identity[i]
        spl = make_smoothing_spline(x, ek, lam=l)
        A[:, i] = spl(x)

    assert np.allclose(A, A.T)
    a_kk = np.diag(A)
    assert (a_kk > 0).all() and (a_kk < 1).all()
    assert 2 < np.trace(A) < n
    trA_all.append(np.trace(A))

    # Then find V(lam), V0(lam)
    RSS = np.sum(np.square(y - y_hat))

    # GCV
    df = np.trace(A) / len(x)
    v_gcv_lam.append((RSS / len(x)) / (1 - df) ** 2)

    # LOOCV
    denom = (1 - np.diag(A)) ** 2
    v_loo_lam.append((1 / len(x)) * np.sum(np.square(y - y_hat) / denom))

i_gcv = np.argmin(v_gcv_lam)
i_loo = np.argmin(v_loo_lam)

delta = np.log10(lam[i_gcv]) - np.log10(lam[i_loo])
r = v_loo_lam[i_gcv] / v_loo_lam[i_loo]

assert 0 < i_gcv < len(lam) - 1 and 0 < i_loo < len(lam) - 1  # interior minima
assert r >= 1

pd.DataFrame(
    {"lam": lam, "trA": trA_all, "V_gcv": v_gcv_lam, "V_loo": v_loo_lam}
).to_csv("run1_scores.csv", index=False)

print("The delta between GCV and LOOCV for t=x case. ", delta)
print("The ratio between GCV and LOOCV for t=x case. ", r)

plt.loglog(lam, v_gcv_lam, label="GCV  V(λ)")
plt.loglog(lam, v_loo_lam, "--", label="LOOCV  V₀(λ)")
plt.axvline(lam[i_gcv], color="C0", ls=":", alpha=0.6)
plt.axvline(lam[i_loo], color="C1", ls=":", alpha=0.6)
plt.xlabel("λ")
plt.ylabel("score")
plt.legend()
plt.title("Run 1: scipy t=None, GCV vs exact LOOCV")
plt.savefig("run1_curves.png", dpi=150)
plt.show()
