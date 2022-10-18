library(rstudioapi)
library(ggplot2)
library(readr)
library(grid)
library(gridExtra)

curdir <- dirname(getSourceEditorContext()$path)
setwd(curdir)


df <- read_csv("estimation_slowdrifts.csv") # all systematic updating weights put to 0
df <- read_csv("estimation_slowdrifts_withSys.csv") # systematic updating present

df_iter <- df[df$dataset==5,]
ggplot(df_iter, aes(x=trial, y=drift_value, col = as.factor(drift_type))) +
  geom_line() +
  theme_bw() +
  xlim(c(0,2000)) +
  labs(x = "Trials", y = "Drift") +
  theme(legend.position="top")
