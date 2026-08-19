# Run 2: R smooth.spline with all.knots=TRUE (t = x), GCV vs exact LOOCV.
# Mirrors gcv-testing.py: same dataset CSV, same lambda grid, same formulas.
#
# Note on lambda: smooth.spline's `lambda` is defined on x rescaled to [0,1],
# so lambda_scipy = lambda_R * (max(x) - min(x))^3. Irrelevant for Delta and r
# (both are within-run quantities), but R's argmin lambda will not numerically
# equal Python's.

d <- read.csv("experiment_dataset.csv")
x <- d$x
y <- d$y
n <- length(x)

lams <- 10^seq(-6, 3, length.out = 100)

v_gcv <- numeric(length(lams))
v_loo <- numeric(length(lams))
trA_all <- numeric(length(lams))

lev_checked <- FALSE

for (j in seq_along(lams)) {
  l <- lams[j]

  fit <- smooth.spline(x, y, all.knots = TRUE, lambda = l)
  y_hat <- predict(fit, x)$y

  # influence matrix, column by column via unit vectors
  A <- matrix(0, n, n)
  for (k in 1:n) {
    ek <- numeric(n)
    ek[k] <- 1
    A[, k] <- predict(smooth.spline(x, ek, all.knots = TRUE, lambda = l), x)$y
  }

  stopifnot(isTRUE(all.equal(A, t(A), tolerance = 1e-8)))
  a_kk <- diag(A)
  stopifnot(all(a_kk > 0), all(a_kk < 1))
  trA <- sum(a_kk)
  stopifnot(trA > 2, trA < n)
  trA_all[j] <- trA

  # freebie check, once: R's own leverages must match the unit-vector diag(A)
  if (!lev_checked) {
    stopifnot(isTRUE(all.equal(a_kk, fit$lev, tolerance = 1e-8)))
    cat("lev check passed at lambda =", l, "\n")
    lev_checked <- TRUE
  }

  res <- y - y_hat
  v_gcv[j] <- mean(res^2) / (1 - trA / n)^2
  v_loo[j] <- mean((res / (1 - a_kk))^2)
}

i_gcv <- which.min(v_gcv)
i_loo <- which.min(v_loo)
stopifnot(i_gcv > 1, i_gcv < length(lams), i_loo > 1, i_loo < length(lams))

delta <- log10(lams[i_gcv]) - log10(lams[i_loo])
r <- v_loo[i_gcv] / v_loo[i_loo]
stopifnot(r >= 1)

cat("Run 2 (R, all.knots=TRUE):\n")
cat("  lambda_GCV =", lams[i_gcv], " lambda_LOO =", lams[i_loo], "\n")
cat("  Delta (decades) =", delta, "\n")
cat("  r = V0(lam_GCV)/V0(lam_LOO) =", r, "\n")
cat("  trA at grid ends:", trA_all[1], trA_all[length(lams)], "\n")

write.csv(data.frame(lam = lams, trA = trA_all, V_gcv = v_gcv, V_loo = v_loo),
          "run2_scores.csv", row.names = FALSE)

png("run2_curves.png", width = 960, height = 720, res = 150)
plot(lams, v_gcv, log = "xy", type = "l", col = "blue",
     xlab = "lambda (R scale)", ylab = "score",
     main = "Run 2: R all.knots=TRUE, GCV vs exact LOOCV")
lines(lams, v_loo, col = "orange", lty = 2)
abline(v = lams[i_gcv], col = "blue", lty = 3)
abline(v = lams[i_loo], col = "orange", lty = 3)
legend("topleft", c("GCV V", "LOOCV V0"), col = c("blue", "orange"), lty = c(1, 2))
dev.off()
