library(dplyr)
library(ranger)
library(tm)
library(caret)
library(tictoc)

setwd("C:/Users/onlooker/OneDrive/DML/data")

text_label = readRDS("TFIDF_label.RData")
text_unlabel = readRDS("TFIDF_unlabel.RData")

covariate_label = read.csv("controls_label.csv")
covariate_unlabel = read.csv("controls_unlabel.csv")


fb_label = read.csv("12000_labeled_sample.txt", stringsAsFactors = FALSE, sep="\t")
fb_label = fb_label |> 
  filter(Irrelevant < 3 & None < 3) |>
  filter(type != "offer") |>
  mutate(sentiment = factor(ifelse(Positive >= 3, 1, 0))) |>
  select(-c(5:12)) |>
  filter(row_number() %in% as.integer(text_label$dimnames$Docs))

fb_unlabel = read.csv("full data.txt", sep = "\t")
fb_unlabel = fb_unlabel |>
  filter(public_id != from_id & from_category == "-") |>
  filter(type != "offer" & type != "swf") |>
  select(id, message, likes_count, comments_count, type) |>
  filter(row_number() %in% as.integer(text_unlabel$dimnames$Docs))


text_label = readRDS("BERT_label.RData")
text_unlabel = readRDS("BERT_unlabel.RData")


control_label = covariate_label |>
  mutate(photo = ifelse(type == "photo", 1, 0),
         status = ifelse(type == "status", 1, 0),
         video = ifelse(type == "video", 1, 0),
         const = 1) |>
  select(wordcount, photo, status, video, const)
sentiment_label = fb_label |> select(sentiment)
outcome_label = fb_label |> 
  select(comments_count) |> 
  mutate(comments_count = log(comments_count + 1))

control_unlabel = covariate_unlabel |> 
  mutate(photo = ifelse(type == "photo", 1, 0),
         status = ifelse(type == "status", 1, 0),
         video = ifelse(type == "video", 1, 0),
         const = 1) |>
  select(wordcount, photo, status, video, const)
sentiment_unlabel = data.frame(sentiment = factor(rep(0, nrow(control_unlabel)), levels = c(0, 1)))
outcome_unlabel = fb_unlabel |> 
  select(comments_count) |> 
  mutate(comments_count = log(comments_count + 1))


control = rbind.data.frame(control_label, control_unlabel)
outcome = rbind.data.frame(outcome_label, outcome_unlabel)

extra_Z = as.matrix(control |> select(wordcount, photo, status, video))
extra_Y = as.matrix(outcome)

text = as.data.frame(rbind(text_label, text_unlabel))
text = cbind(text, extra_Z, extra_Y)
sentiment = rbind.data.frame(sentiment_label, sentiment_unlabel)


## (unbiased) linear regression

dat = data.frame(outcome_label, sentiment_label, covariate_label)
model_lm = lm(comments_count ~ sentiment + wordcount + type, data = dat)
summary(model_lm)

## DML method
set.seed(27101)

N = nrow(control)
p = 6
N_label = nrow(control_label)
N_unlabel = nrow(control_unlabel)

label = c(rep(1, N_label), rep(0, N_unlabel))
label_index = seq(1, N_label, 1)

K = 5
folding = rep(0, N)
folding[label_index] = sample(rep(1:K, length.out = N_label), N_label, replace = FALSE)
folding[-label_index] = sample(rep(1:K, length.out = N_unlabel), N_unlabel, replace = FALSE)  

PSIa = matrix(0, nrow = p, ncol = p)
PSIb = matrix(0, nrow = p, ncol = 1)  

for(k in 1:K)
{
  set.seed(101)
  text_folding = text[folding == k, ]
  sentiment_folding = sentiment[folding == k, ]
  text_other_obs = text[folding != k & label == 1, ]
  sentiment_other_obs = sentiment[folding != k & label == 1, ]
  
  lambda = mean(label[folding != k])
  
  ntrees = 100
  model_mu1 = ranger(x = text_other_obs, y = sentiment_other_obs, num.trees = ntrees, probability = T)
  pred_mu1 = predict(model_mu1, text_folding, type="response")
  mu1 = pred_mu1$predictions[, 2, drop = FALSE]
  mu2 = mu1
  
  Y = as.matrix(outcome[folding == k, ])
  X = as.matrix(as.numeric(sentiment_folding) - 1)
  R = as.matrix(label[folding == k])
  Z = as.matrix(control[folding == k,])
  
  PSIa = PSIa + rbind(cbind(sum(R * (X ^ 2 - mu2) / lambda + mu2), t(R * (X - mu1) / lambda + mu1) %*% Z), 
                      cbind(t(Z) %*% (R * (X - mu1) / lambda + mu1), t(Z) %*% Z)) / N
  PSIb = PSIb + rbind(sum(Y * (R * (X - mu1) / lambda + mu1)), t(Z) %*% Y) / N
}
Coef=solve(PSIa,PSIb)  


J0 = PSIa
BETA = Coef[1, 1]
GAMMA = Coef[-1, , drop = FALSE]
PSI2 = matrix(0, nrow = p, ncol = p)
for(k in 1:K)
{
  set.seed(101)
  text_folding = text[folding == k, ]
  sentiment_folding = sentiment[folding == k, ]
  text_other_obs = text[folding != k & label == 1, ]
  sentiment_other_obs = sentiment[folding != k & label == 1, ]
  
  lambda = mean(label[folding != k])
  
  ntrees = 100
  model_mu1 = ranger(x = text_other_obs, y = sentiment_other_obs, num.trees = ntrees, probability = T)
  pred_mu1 = predict(model_mu1, text_folding, type="response")
  mu1 = pred_mu1$predictions[, 2, drop = FALSE]
  mu2 = mu1
  
  Y = as.matrix(outcome[folding == k, ])
  X = as.matrix(as.numeric(sentiment_folding) - 1)
  R = as.matrix(label[folding == k])
  Z = as.matrix(control[folding == k,])
  
  PSI=sapply(1:length(sentiment_folding), function(i){rbind((R[i,1]*(X[i,1]-mu1[i,1])/lambda+mu1[i,1])*(Y[i,1]-Z[i,,drop=F]%*%GAMMA)-(R[i,1]*(X[i,1]^2-mu2[i,1])/lambda+mu2[i,1])*BETA,
                                                            t(Z[i,,drop=F])%*%(Y[i,1]-(R[i,1]*(X[i,1]-mu1[i,1])/lambda+mu1[i,1])*BETA-Z[i,,drop=F]%*%GAMMA))})
  PSI2=PSI2+PSI%*%t(PSI)/N
}

sigma2=solve(J0)%*%PSI2%*%t(solve(J0))/N  


print(Coef)
print(sqrt(diag(sigma2)))

