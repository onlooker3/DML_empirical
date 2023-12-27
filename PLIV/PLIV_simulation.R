library(tidyverse)
library(mlr3verse)
library(DoubleML)
library(ivreg)
library(furrr)
library(purrr)
library(tictoc)

size = c(100, 500, 1000, 5000, 10000)
K = 5
num_iter = 100

PLIV_LIV = function(seed, N)
{
  X1 = rnorm(N, 0, 2)
  X2 = rnorm(N, 0, 2)
  X = data.frame(X1 = X1, X2 = X2)
  
  Z0 = rbinom(N, 1, 1 / (1 + exp(-(X1 + X2 + 1))))
  D = Z0 + runif(N, -0.5, 0.5)
  Z = Z0 + runif(N, -0.5, 0.5)
  Y = Z0 + X1 + X2 + X1 * X2 + X1 ^ 2 + X2 ^ 2 + 1 + rnorm(N, 0, 1)
  dat = data.frame(Y = Y, D = D, Z = Z, X)
  dml_data = DoubleMLData$new(dat, y_col="Y", d_col = "D", z_cols= "Z")
  
  # 2SLS
  model_2SLS = ivreg(Y ~ D + X1 + X2 | Z + X1 + X2, data = dat)
  model_2SLS$coefficients
  
  # DML with random forest
  lr_ranger = lrn("regr.ranger")
  ml_l_ranger = lr_ranger$clone()
  ml_m_ranger = lr_ranger$clone()
  ml_r_ranger = lr_ranger$clone()
  dml_ranger = DoubleMLPLIV$new(dml_data, ml_l_ranger, ml_m_ranger, ml_r_ranger)
  dml_ranger$fit()
  
  # DML with xgboost
  lr_xgboost = lrn("regr.xgboost", nrounds = 100, early_stopping_set = "test")
  ml_l_xgboost = lr_xgboost$clone()
  ml_m_xgboost = lr_xgboost$clone()
  ml_r_xgboost = lr_xgboost$clone()
  dml_xgboost = DoubleMLPLIV$new(dml_data, ml_l_xgboost, ml_m_xgboost, ml_r_xgboost)
  dml_xgboost$fit()
  
  list(theta_ranger = dml_ranger$coef, theta_xgboost = dml_xgboost$coef, theta_LIV = model_2SLS$coefficients['D'])
}

tic()

plan(multisession, workers = 16)
set.seed(101)
x_seq = rep(1:num_iter, times = length(size))
y_seq = rep(size, each = num_iter)
theta = future_map2_dfr(x_seq, y_seq, PLIV_LIV, .options = furrr_options(seed = T))

theta_ranger = matrix(theta$theta_ranger, nrow = num_iter, ncol = length(size))
theta_xgboost = matrix(theta$theta_xgboost, nrow = num_iter, ncol = length(size))
theta_LIV = matrix(theta$theta_LIV, nrow = num_iter, ncol = length(size))

mean_ranger = apply(theta_ranger, 2, mean)
mean_xgboost = apply(theta_xgboost, 2, mean)
mean_LIV = apply(theta_LIV, 2, mean)
sd_ranger = apply(theta_ranger, 2, sd)
sd_xgboost = apply(theta_xgboost, 2, sd)
sd_LIV = apply(theta_LIV, 2, sd)

mean_ranger
sd_ranger

mean_xgboost
sd_xgboost

mean_LIV
sd_LIV

toc()


