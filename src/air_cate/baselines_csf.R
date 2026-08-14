#!/usr/bin/env Rscript
#
# Fit a grf causal survival forest once per horizon and predict tau(t|x) on a test set.
#
# With target = "survival.probability" and horizon = h, causal_survival_forest
# estimates the scalar P[T(1) > h | X] - P[T(0) > h | X], which is exactly the
# time-fixed CATE tau_h(x). Recovering the whole tv-CATE curve therefore takes one
# fit per point of the time grid. That loop lives here rather than in Python so a
# replication costs one R process instead of one per horizon.
#
# Usage: Rscript baselines_csf.R <workdir> <num_trees> <num_threads> <seed>
#
# Reads  <workdir>/{train,test,horizons}.csv
# Writes <workdir>/predictions.csv  -- n_test rows x length(horizons) columns,
#        NaN in any column whose horizon could not be fit.

suppressPackageStartupMessages(library(grf))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L) {
  stop("usage: baselines_csf.R <workdir> <num_trees> <num_threads> <seed>")
}
workdir     <- args[[1]]
num_trees   <- as.integer(args[[2]])
num_threads <- as.integer(args[[3]])
seed        <- as.numeric(args[[4]])

train    <- read.csv(file.path(workdir, "train.csv"))
test     <- read.csv(file.path(workdir, "test.csv"))
horizons <- read.csv(file.path(workdir, "horizons.csv"))$horizon

# Covariates are every train column that is not one of the three outcome/treatment
# columns; test.csv carries the covariates alone, in the same order.
outcome_cols <- c("W", "Z", "Delta")
x_cols <- setdiff(names(train), outcome_cols)
if (!identical(x_cols, names(test))) {
  stop(sprintf("covariate mismatch: train has (%s), test has (%s)",
               paste(x_cols, collapse = ","), paste(names(test), collapse = ",")))
}

X      <- as.matrix(train[, x_cols, drop = FALSE])
X_test <- as.matrix(test[, x_cols, drop = FALSE])
W      <- train$W
Y      <- train$Z
D      <- train$Delta

preds <- matrix(NaN, nrow = nrow(X_test), ncol = length(horizons))
n_failed <- 0L

for (j in seq_along(horizons)) {
  h <- horizons[[j]]
  # grf refuses a horizon past the last observed failure time, and can fail when a
  # horizon has too few events to split on. Both are expected in the tail of small
  # samples, so a failure is left as NaN and repaired on the Python side (flat
  # tail) instead of killing the whole replication.
  ok <- tryCatch({
    csf <- causal_survival_forest(
      X, Y = Y, W = W, D = D,
      target      = "survival.probability",
      horizon     = h,
      num.trees   = num_trees,
      num.threads = num_threads,
      # Distinct per horizon so the forests are not identically correlated, but
      # still a deterministic function of the replication seed. The modulus keeps
      # it inside R's 32-bit integer range for any grid size.
      seed        = as.integer((seed + j) %% 2147483647)
    )
    preds[, j] <- predict(csf, newdata = X_test)$predictions
    TRUE
  }, error = function(e) {
    message(sprintf("horizon %.6g failed: %s", h, conditionMessage(e)))
    FALSE
  })
  if (!ok) n_failed <- n_failed + 1L
}

write.csv(preds, file.path(workdir, "predictions.csv"), row.names = FALSE)
message(sprintf("fitted %d/%d horizons (num.trees=%d, num.threads=%d)",
                length(horizons) - n_failed, length(horizons),
                num_trees, num_threads))
