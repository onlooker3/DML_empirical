library(tidyverse)
library(mlr3verse)
library(DoubleML)
library(Matching)
library(xgboost)
library(furrr)
library(purrr)
library(tictoc)

size = c(100, 500, 1000, 5000, 10000)
K = 5
num_iter = 100

PLR_lm = function(seed, N)
{
  # data generation process
  X1 = rnorm(N, 0, 2)
  X2 = rnorm(N, 0, 2)
  D = rbinom(N, 1, 1 / (1 + exp(-(X1 + X2 + 1))))
  Y = D + X1 + X2 + X1 * X2 + X1 ^ 2 + X2 ^ 2 + 1 + rnorm(N, 0, 1)
  dat = data.frame(Y = Y, D = D, X1 = X1, X2 = X2)
  dml_data = DoubleMLData$new(dat, y_col="Y", d_col = "D")
  
  # Linear regression
  model_lm = lm(Y ~ D + X1 + X2, data = dat)
  
  # Propensity score matching (logit, M = 1)
  PSM_logit = glm(D ~ X1 + X2, data = dat, family = binomial(link = "logit"))
  model_logit = Match(Y = dat$Y, Tr = dat$D, X = PSM_logit$fitted.values, Z = dat[,c('X1', 'X2')], BiasAdjust = T, estimand = 'ATE', M = 1)
  #model_logit = Match(Y = dat$Y, Tr = dat$D, X = PSM_logit$fitted.values, Z = dat[,c('X1', 'X2')], BiasAdjust = T, estimand = 'ATE', M = 5)
  
  # Propensity score matching (probit, M = 1)
  PSM_probit = glm(D ~ X1 + X2, data = dat, family = binomial(link = "probit"))
  model_probit = Match(Y = dat$Y, Tr = dat$D, X = PSM_probit$fitted.values, Z = dat[,c('X1', 'X2')], BiasAdjust = T, estimand = 'ATE', M = 1)
  #model_probit = Match(Y = dat$Y, Tr = dat$D, X = PSM_probit$fitted.values, Z = dat[,c('X1', 'X2')], BiasAdjust = T, estimand = 'ATE', M = 5)
  
  # IPW with regression adjustment (logit)
  PS_logit = glm(D ~ X1 + X2, data = dat, family = binomial(link = "logit"))
  logit_fit = PS_logit$fitted.values
  logit_weight = 1 / (D * logit_fit + (1 - D) * (1 - logit_fit))
  model_IPW_logit = lm(Y ~ D + X1 + X2, data = dat, weights = logit_weight)
  
  # IPW with regression adjustment (probit)
  PS_probit = glm(D ~ X1 + X2, data = dat, family = binomial(link = "probit"))
  probit_fit = PS_probit$fitted.values
  probit_weight = 1 / (D * probit_fit + (1 - D) * (1 - probit_fit))
  model_IPW_probit = lm(Y ~ D + X1 + X2, data = dat, weights = probit_weight)
  
  # DML with random forest
  ml_l_ranger = lrn("regr.ranger")
  ml_m_ranger = lrn("classif.ranger")
  dml_ranger = DoubleMLPLR$new(dml_data, ml_l_ranger, ml_m_ranger)
  dml_ranger$fit()

  # DML with xgboost
  ml_l_xgboost = lrn("regr.xgboost", nrounds = 100, early_stopping_set = "test")
  ml_m_xgboost = lrn("classif.xgboost", nrounds = 100, early_stopping_set = "test")
  dml_xgboost = DoubleMLPLR$new(dml_data, ml_l_xgboost, ml_m_xgboost)
  dml_xgboost$fit()
  
  list(theta_ranger = dml_ranger$coef, theta_xgboost = dml_xgboost$coef, 
       theta_lm = model_lm$coefficients['D'], theta_logit = model_logit$est, theta_probit = model_probit$est,
       theta_IPW_logit = model_IPW_logit$coefficients['D'], theta_IPW_probit = model_IPW_probit$coefficients['D'])
}

tic()

plan(multisession, workers = 16)
set.seed(101)
x_seq = rep(1:num_iter, times = length(size))
y_seq = rep(size, each = num_iter)
theta = future_map2_dfr(x_seq, y_seq, PLR_lm, .options = furrr_options(seed = T))

theta_ranger = matrix(theta$theta_ranger, nrow = num_iter, ncol = length(size))
theta_xgboost = matrix(theta$theta_xgboost, nrow = num_iter, ncol = length(size))
theta_lm = matrix(theta$theta_lm, nrow = num_iter, ncol = length(size))
theta_logit = matrix(theta$theta_logit, nrow = num_iter, ncol = length(size))
theta_probit = matrix(theta$theta_probit, nrow = num_iter, ncol = length(size))
theta_IPW_logit = matrix(theta$theta_IPW_logit, nrow = num_iter, ncol = length(size))
theta_IPW_probit = matrix(theta$theta_IPW_probit, nrow = num_iter, ncol = length(size))

mean_ranger = apply(theta_ranger, 2, mean)
mean_xgboost = apply(theta_xgboost, 2, mean)
mean_lm = apply(theta_lm, 2, mean)
mean_logit = apply(theta_logit, 2, mean)
mean_probit = apply(theta_probit, 2, mean)
mean_IPW_logit = apply(theta_IPW_logit, 2, mean)
mean_IPW_probit = apply(theta_IPW_probit, 2, mean)

sd_ranger = apply(theta_ranger, 2, sd)
sd_xgboost = apply(theta_xgboost, 2, sd)
sd_lm = apply(theta_lm, 2, sd)
sd_logit = apply(theta_logit, 2, sd)
sd_probit = apply(theta_probit, 2, sd)
sd_IPW_logit = apply(theta_IPW_logit, 2, sd)
sd_IPW_probit = apply(theta_IPW_probit, 2, sd)

mean_ranger
sd_ranger

mean_xgboost
sd_xgboost

mean_lm
sd_lm

mean_logit
sd_logit

mean_probit
sd_probit

mean_IPW_logit
sd_IPW_logit

mean_IPW_probit
sd_IPW_probit

toc()


