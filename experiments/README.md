# GCV vs exact LOOCV: does t != x change anything?

Experiment for the GCV follow-up to scipy PR
[#25862](https://github.com/scipy/scipy/pull/25862) (`t=` in
`make_smoothing_spline`). Question (proposed by Evgeni Burovski): GCV is
trusted for knots at the data (t = x); does its agreement with exact
leave-one-out cross-validation change when the knots are user-chosen
(t != x)?

## Method

One fixed dataset for every run: `experiment_dataset.csv`, n = 100,
x sorted uniform on [0, 4], y = sin(x) + N(0, 0.3^2), seed 42.

For each lambda on a log grid (1e-6 to 1e3, 100 points):

- fit once with y, giving fitted values yhat;
- build the influence matrix A column by column: fit the same problem
  with each unit vector e_k as data; since yhat = A y, the output is
  column k of A (in run 2 this construction is checked once against R's
  own `fit$lev` leverages);
- GCV score `V = mean(res^2) / (1 - tr(A)/n)^2`;
- exact LOOCV score `V0 = mean((res / (1 - diag(A)))^2)` (the
  Craven & Wahba leverage identity, no refits).

Per run, two numbers compare the criteria:

- `Delta = log10(lambda_GCV) - log10(lambda_LOO)`, the gap between the
  two minimizers in decades (grid resolution: 9/99 = 0.09 decades);
- `r = V0(lambda_GCV) / V0(lambda_LOO) >= 1`, the leave-one-out error
  paid for using GCV's choice instead of LOOCV's.

Sanity checks asserted in every run: A symmetric, diag(A) in (0, 1),
tr(A) decreasing from the basis size toward 2, minima interior to the
grid, r >= 1.

## Runs and results

| run | fitter                       | knots              | Delta (decades) | r       |
|-----|------------------------------|--------------------|-----------------|---------|
| 1   | scipy `t=None`               | t = x              | 0.09 (1 step)   | 1.00008 |
| 2   | R `smooth.spline`, all knots | t = x              | 0               | 1       |
| 3   | R `smooth.spline`, defaults  | ~35 knots, t != x  | 0               | 1       |
| 4   | scipy `t=` (PR branch)       | 37 basis, t != x   | 0.09 (1 step)   | 1.00008 |

Runs 1-2 are the control (t = x, where GCV is trusted); runs 3-4 are the
question. Conclusion: **the GCV-vs-LOOCV gap does not change when the
knots leave the data**, in either ecosystem, at the resolution of the
grid. This is what the theory predicts (neither criterion's construction
uses the knot vector; Golub, Heath & Wahba 1979 derive GCV for an
arbitrary rectangular design matrix), and here it is measured.

Note: R's `lambda` is defined on x rescaled to [0, 1], so its argmin
differs from scipy's by range(x)^3 = 64; Delta and r are within-run
quantities and unaffected. (Cross-check: R picks 0.01, 0.01 * 64 = 0.64,
scipy's minimum is at ~0.6.)

## Files

- `run1_scipy_t_none.py`, `run4_scipy_user_knots.py` — Python runs
  (run 4 needs the PR branch, e.g. `spin python`).
- `run2_R_all_knots.R`, `run3_R_default_knots.R` — R runs
  (`Rscript run2_R_all_knots.R`).
- `experiment_dataset.csv` — the shared dataset (regenerate only
  deliberately; all runs read this file).
- `run{1,2,3,4}_scores.csv` — per-lambda tables (lambda, trA, V, V0).
- `run{1,2,3,4}_curves.png` — overlay plots of the two criteria.

Reproduction: Python 3.12, scipy PR branch `userknots`; R 4.x,
stats::smooth.spline.
