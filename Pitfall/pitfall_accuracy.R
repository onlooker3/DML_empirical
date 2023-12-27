library(ggplot2)
library(tidyverse)

setwd("C:/Users/onlooker/OneDrive/DML/simulation/Pitfall/pitfall")

dat_rf = t(read.table(file = "result_rf.txt", header = F))
dat_xgboost = t(read.table(file = "result_xgboost.txt", header = F))
dat_tree = t(read.table(file = "result_tree.txt", header = F))

dat_all = rbind(dat_rf, dat_xgboost, dat_tree)


label = c(rep("random forest", nrow(dat_rf)), rep("xgboost", nrow(dat_xgboost)), rep("decision tree", nrow(dat_tree)))
dat = data.frame(dat_all, label = label)
colnames(dat) = c("pred_error", "est_error", "label")

pdf("pitfall_plot.pdf",10,6)
ggplot(data = dat |> filter(est_error < 2)) + 
  geom_point(aes(x = pred_error, y = est_error, shape = label, color = label), size = 3) +
  xlab("prediction error") + ylab("estimation error") +
  guides(shape = guide_legend(override.aes = list(size = 4))) +
  theme_bw() + theme(text = element_text(size=21),
                     panel.grid.major = element_blank(), 
                     panel.grid.minor = element_blank(),
                     panel.background = element_blank())
dev.off()
