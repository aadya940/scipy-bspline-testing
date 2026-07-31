import numpy as np
from scipy.interpolate import BSpline
from scipy.integrate import quad

# Validation companion for "An Exact Penalty Matrix for Cubic Smoothing
# Splines on Arbitrary Knot Vectors". Section and equation numbers below
# refer to the report. Every check is an assert; the script passing means
# all checks passed.

# Running example of the report: irregular breaks, clamped ends.
breaks = np.array([0., 0.7, 2.3, 3.])
tau = np.concatenate([np.repeat(breaks[0], 4), breaks[1:-1], np.repeat(breaks[-1], 4)])
m = len(tau) - 4    # number of cubic basis functions (Sec. 2)


# Sec. 4: differentiation as a matrix 

def deriv_matrix(tau, order):
    # eq. (5): row j has -d_j at column j-1 and +d_j at column j,
    # d_j = (order-1) / (tau[j+order-1] - tau[j]).
    # Zero denominator = repeated boundary knots: the target basis
    # function is identically zero there, so the row is left zero.
    N = len(tau) - order
    D = np.zeros((N + 1, N))
    for j in range(N + 1):
        denom = tau[j + order - 1] - tau[j]
        d = (order - 1) / denom if denom > 0.0 else 0.0
        if j < N:
            D[j, j] = d
        if j >= 1:
            D[j, j - 1] = -d
    return D

D1 = deriv_matrix(tau, 4)   # cubic -> quadratic coefficients
D2 = deriv_matrix(tau, 3)   # quadratic -> linear coefficients

assert D1.shape == (m + 1, m)
assert D2.shape == (m + 2, m + 1)
assert np.allclose(D1[0], 0) and np.allclose(D1[-1], 0)   # clamped ends

C = D2 @ D1                 # eq. (C): coefficients of f'' in the hat basis
assert C.shape == (m + 2, m)


# Sec. 5: mass matrix of the hat functions

# Closed form, eqs. (Rdiag) and (Roff): diagonal is the hat's width over 3,
# off-diagonal the shared interval over 6. Zero-width windows at the
# clamped ends come out zero with no special-casing.
n_hats = len(tau) - 2
R = np.zeros((n_hats, n_hats))
for p in range(n_hats):
    R[p, p] = (tau[p + 2] - tau[p]) / 3.0
    if p + 1 < n_hats:
        R[p, p + 1] = R[p + 1, p] = (tau[p + 2] - tau[p + 1]) / 6.0

# Check the closed form against brute-force quadrature of the same
# integrals, using scipy's own linear B-splines.
def hat(p):
    coef = np.zeros(n_hats)
    coef[p] = 1.0
    return BSpline(tau, coef, k=1, extrapolate=False)

R_quad = np.zeros_like(R)
for p in range(n_hats):
    Np = hat(p)
    for q in range(p, n_hats):
        Nq = hat(q)
        val, _ = quad(lambda x: np.nan_to_num(Np(x)) * np.nan_to_num(Nq(x)),
                      tau[0], tau[-1], limit=200)
        R_quad[p, q] = R_quad[q, p] = val

np.testing.assert_allclose(R, R_quad, atol=1e-10)

# Row-sum identity from Sec. 5: hats sum to one, so each row of R sums
# to half the hat's width.
np.testing.assert_allclose(R.sum(axis=1), (tau[2:] - tau[:-2]) / 2, atol=1e-12)
 
# Sec. 6: the penalty matrix, eq. (omega-crc)

Omega = C.T @ R @ C

# Structure: symmetric, bandwidth 3, and the null space is the straight
# lines, constants (Omega @ 1 = 0) and the linear function, whose
# B-spline coefficients are the Greville points.
assert np.allclose(Omega, Omega.T)
for i in range(m):
    for j in range(m):
        if abs(i - j) > 3:
            assert Omega[i, j] == 0.0
greville = np.array([tau[i + 1:i + 4].mean() for i in range(m)])
np.testing.assert_allclose(Omega @ np.ones(m), 0, atol=1e-12)
np.testing.assert_allclose(Omega @ greville, 0, atol=1e-12)

# 1: output of fda::bsplinepen (R package by Ramsay & Silverman),
# generated with the snippet at the bottom of this file. Only the numbers
# were used; the GPL fda source was not consulted.
OMEGA_FDA = np.array([
    [ 34.985423, -44.012852,   7.785193,   1.242236,   0.      ,   0.      ],
    [-44.012852,  56.280897, -11.140875,  -1.470223,   0.343052,   0.      ],
    [  7.785193, -11.140875,   4.226890,  -0.643222,  -1.470223,   1.242236],
    [  1.242236,  -1.470223,  -0.643222,   4.226890, -11.140875,   7.785193],
    [  0.      ,   0.343052,  -1.470223, -11.140875,  56.280897, -44.012852],
    [  0.      ,   0.      ,   1.242236,   7.785193, -44.012852,  34.985423],
])
np.testing.assert_allclose(Omega, OMEGA_FDA, atol=1e-6)

# 2: Wand & Ormerod (2008), eq. (3). Each B_i'' B_j'' is piecewise
# quadratic, so Simpson's rule over each knot interval is exact:
# Omega = Btilde'' diag(w) Btilde'' with evaluations at interval endpoints
# and midpoints. Independent route to the same matrix.
def bpp(u):
    # values of all second derivatives B_j''(u), via scipy
    out = np.empty(m)
    for j in range(m):
        coef = np.zeros(m)
        coef[j] = 1.0
        out[j] = BSpline(tau, coef, k=3)(u, nu=2)
    return out

Omega_wo = np.zeros((m, m))
for a, b in zip(tau[:-1], tau[1:]):
    if b <= a:
        continue
    h = b - a
    for u, w in [(a, h / 6), ((a + b) / 2, 4 * h / 6), (b - 1e-12, h / 6)]:
        v = bpp(u)
        Omega_wo += w * np.outer(v, v)
np.testing.assert_allclose(Omega, Omega_wo, atol=1e-8)

print("all checks passed")

from scipy.interpolate import make_smoothing_spline

# same lam, knots = clamped data sites: the new path minimizes over a
# larger space, but the representer theorem puts the optimum in the
# natural-spline subspace, so the two paths must return the same function.
x = np.linspace(10)
y = 3 * x ** 10 + 4 * x ** 2 + 5

t_x = np.r_[[x[0]] * 4, x[1:-1], [x[-1]] * 4]
for lam in [1e-4, 1e-2, 1.0, 100.0]:
    s_old = make_smoothing_spline(x, y, lam=lam)
    s_new = make_smoothing_spline(x, y, lam=lam, t=t_x)
    np.testing.assert_allclose(s_new(grid), s_old(grid), atol=1e-8)

# fda generation (R). Run separately; paste output into OMEGA_FDA.
#
#   library(fda)
#   basisobj <- create.bspline.basis(rangeval = c(0, 3),
#                                    breaks = c(0, 0.7, 2.3, 3), norder = 4)
#   print(round(bsplinepen(basisobj, Lfdobj = 2), 6))
