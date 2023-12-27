library(tidyverse)
library(mlr3verse)
library(gridExtra)
library(DoubleML)
library(Matching)
library(xgboost)
library(furrr)
library(purrr)
library(tictoc)

N = 1000
K = 5
num_iter = 1000

non_orth_score = function(y, d, l_hat, m_hat, g_hat, smpls)
{
  u_hat = y - g_hat
  psi_a = -1*d*d
  psi_b = d*u_hat
  psis = list(psi_a = psi_a, psi_b = psi_b)
  return(psis)
}

PLR_lm = function(N)
{
  x1 = rnorm(N, 0, 2)
  x2 = rnorm(N, 0, 2)
  
  d = rbinom(N, 1, 1 / (1 + exp(-(x1 + x2 + 1))))
  y = d + x1 + x2 + x1 * x2 + x1 ^ 2 + x2 ^ 2 + 1 + rnorm(N, 0, 1)
  dat = data.frame(y = y, d = d, x1 = x1, x2 = x2)
  dml_data = DoubleMLData$new(dat, y_col = "y", d_col = "d")
  
  # Linear regression
  model_lm = lm(y ~ d + x1 + x2, data = dat)
  
  # Naive estimator with non-orthogonal score
  ml_l = lrn("regr.ranger")
  ml_m = lrn("classif.ranger")
  ml_g = ml_l$clone()
  model_naive = DoubleMLPLR$new(dml_data, ml_l, ml_m, ml_g, score = non_orth_score,
                                n_folds = 1, apply_cross_fitting = FALSE)
  model_naive$fit()
  
  # DML without cross-fitting
  ml_l = lrn("regr.ranger")
  ml_m = lrn("classif.ranger")
  model_nocross = DoubleMLPLR$new(dml_data, ml_l, ml_m, n_folds = 1, apply_cross_fitting = FALSE)
  model_nocross$fit()
  
  # DML with random forest
  ml_l = lrn("regr.ranger")
  ml_m = lrn("classif.ranger")
  model_dml = DoubleMLPLR$new(dml_data, ml_l, ml_m, n_folds = 5, apply_cross_fitting = TRUE)
  model_dml$fit()
  
  list(theta_lm = model_lm$coefficients['d'], theta_naive = model_naive$coef, theta_nocross = model_nocross$coef, theta_dml = model_dml$coef,
       se_lm = sqrt(diag(vcov(model_lm)))['d'], se_naive = model_naive$se, se_nocross = model_nocross$se, se_dml = model_dml$se)
}

plan(multisession, workers = 10)
set.seed(101)

theta = future_map_dfr(rep(N, each = num_iter), PLR_lm, .options = furrr_options(seed = T))




g_lm = ggplot(data.frame(theta_rescaled=(theta$theta_lm - 1)/theta$se_lm)) +
  geom_histogram(aes(y=after_stat(density), x=theta_rescaled, colour = "Simulation", fill="Simulation"), bins = 50, alpha = 1) +
  geom_vline(aes(xintercept = 0), col = "black") + suppressWarnings(geom_function(fun = dnorm, aes(colour = "N(0, 1)", fill="N(0, 1)"))) +
  scale_color_manual(name='', breaks=c("Simulation", "N(0, 1)"), values=c("Simulation"="lightblue", "N(0, 1)"="black")) +
  scale_fill_manual(name='', breaks=c("Simulation", "N(0, 1)"), values=c("Simulation"="lightblue", "N(0, 1)"= "white")) +
  xlim(c(-12, 6)) + xlab("") + ylab("") + ggtitle("Linear regression, N = 1000") + 
  theme_bw() + theme(legend.position=c(0.15, 0.85), legend.key.size = unit(25, "pt"), legend.text = element_text(size = 11), 
                     plot.title = element_text(size=16,hjust=0.5,face="bold"), axis.text=element_text(size=15),
                     panel.grid.major =element_blank(), 
                     panel.grid.minor = element_blank(),
                     panel.background = element_blank())
g_lm


g_naive = ggplot(data.frame(theta_rescaled=(theta$theta_naive - 1)/theta$se_naive)) +
  geom_histogram(aes(y=after_stat(density), x=theta_rescaled, colour = "Simulation", fill="Simulation"), bins = 50, alpha = 1) +
  geom_vline(aes(xintercept = 0), col = "black") + suppressWarnings(geom_function(fun = dnorm, aes(colour = "N(0, 1)", fill="N(0, 1)"))) +
  scale_color_manual(name='', breaks=c("Simulation", "N(0, 1)"), values=c("Simulation"="lightblue", "N(0, 1)"="black")) +
  scale_fill_manual(name='', breaks=c("Simulation", "N(0, 1)"), values=c("Simulation"="lightblue", "N(0, 1)"= "white")) +
  xlim(c(-12, 6)) + xlab("") + ylab("") + ggtitle("Non-Orthogonal, N = 1000") + 
  theme_bw() + theme(legend.position=c(0.15, 0.85), legend.key.size = unit(25, "pt"), legend.text = element_text(size = 11), 
                     plot.title = element_text(size=16,hjust=0.5,face="bold"), axis.text=element_text(size=15),
                     panel.grid.major =element_blank(), 
                     panel.grid.minor = element_blank(),
                     panel.background = element_blank())
g_naive


g_nocross = ggplot(data.frame(theta_rescaled=(theta$theta_nocross - 1)/theta$se_nocross)) +
  geom_histogram(aes(y=after_stat(density), x=theta_rescaled, colour = "Simulation", fill="Simulation"), bins = 50, alpha = 1) +
  geom_vline(aes(xintercept = 0), col = "black") + suppressWarnings(geom_function(fun = dnorm, aes(colour = "N(0, 1)", fill="N(0, 1)"))) +
  scale_color_manual(name='', breaks=c("Simulation", "N(0, 1)"), values=c("Simulation"="lightblue", "N(0, 1)"="black")) +
  scale_fill_manual(name='', breaks=c("Simulation", "N(0, 1)"), values=c("Simulation"="lightblue", "N(0, 1)"= "white")) +
  xlim(c(-12, 6)) + xlab("") + ylab("") + ggtitle("Without cross-fitting, N = 1000") + 
  theme_bw() + theme(legend.position=c(0.15, 0.85), legend.key.size = unit(25, "pt"), legend.text = element_text(size = 11), 
                     plot.title = element_text(size=16,hjust=0.5,face="bold"), axis.text=element_text(size=15),
                     panel.grid.major =element_blank(), 
                     panel.grid.minor = element_blank(),
                     panel.background = element_blank())
g_nocross


g_dml = ggplot(data.frame(theta_rescaled=(theta$theta_dml - 1)/theta$se_dml)) +
  geom_histogram(aes(y=after_stat(density), x=theta_rescaled, colour = "Simulation", fill="Simulation"), bins = 50, alpha = 1) +
  geom_vline(aes(xintercept = 0), col = "black") + suppressWarnings(geom_function(fun = dnorm, aes(colour = "N(0, 1)", fill="N(0, 1)"))) +
  scale_color_manual(name='', breaks=c("Simulation", "N(0, 1)"), values=c("Simulation"="lightblue", "N(0, 1)"="black")) +
  scale_fill_manual(name='', breaks=c("Simulation", "N(0, 1)"), values=c("Simulation"="lightblue", "N(0, 1)"= "white")) +
  xlim(c(-12, 6)) + xlab("") + ylab("") + ggtitle("Double Machine Learning, N = 1000") + 
  theme_bw() + theme(legend.position=c(0.15, 0.85), legend.key.size = unit(25, "pt"), legend.text = element_text(size = 11), 
                     plot.title = element_text(size=16,hjust=0.5,face="bold"), axis.text=element_text(size=15),
                     panel.grid.major =element_blank(), 
                     panel.grid.minor = element_blank(),
                     panel.background = element_blank())
g_dml


pdf("C:/Users/onlooker/OneDrive/DML/simulation/PLR/PLR_plot.pdf",20,5)
grid.arrange(g_lm,g_naive,g_nocross,g_dml,ncol=4)
dev.off()



