
library(rstudioapi)
library(ggplot2)
library(readr)
library(grid)
library(gridExtra)

curdir <- dirname(getSourceEditorContext()$path)
setwd(curdir)

df <- read_csv("parameter_recovery_inflating_sigma.csv")

p_sens <- ggplot(df, aes(x=as.factor(ntrials), y=sens_fit, col = as.factor(ntrials))) +
  geom_point(position = position_dodge(0.1)) +
  theme_bw() +
  geom_hline(yintercept=10) +
  xlab("n_trials") +
  stat_summary(geom = "point",fun = "mean",col = "black",size = 3,shape = 24,fill = "red") +
  #labs(col = "n_iterations", x = "sens_sim") +
  theme(legend.position="none")

p_bias <- ggplot(df, aes(x=as.factor(ntrials), y=bias_fit, col = as.factor(ntrials))) +
  geom_point(position = position_dodge(.1)) +
  theme_bw() +
  geom_hline(yintercept=-5) +
  xlab("n_trials") +
  stat_summary(geom = "point",fun = "mean",col = "black",size = 3,shape = 24,fill = "red") +
  #labs(col = "n_iterations", x = "bias_sim") +
  theme(legend.position="none")

p_sigma <- ggplot(df, aes(x=as.factor(ntrials), y=sigma_fit, col = as.factor(ntrials))) +
  geom_point(position = position_dodge(.1)) +
  theme_bw() +
  geom_hline(yintercept=.05) +
  xlab("n_trials") +
  stat_summary(geom = "point",fun = "mean",col = "black",size = 3,shape = 24,fill = "red") +
  theme(legend.position="none")

p_pc <- ggplot(df, aes(x=as.factor(ntrials), y=pc_fit, col = as.factor(ntrials))) +
  geom_point(position = position_dodge(.1)) +
  theme_bw() +
  geom_hline(yintercept=1) +
  xlab("n_trials") +
  stat_summary(geom = "point",fun = "mean",col = "black",size = 3,shape = 24,fill = "red") +
  theme(legend.position="none")

p_pe <- ggplot(df, aes(x=as.factor(ntrials), y=pe_fit, col = as.factor(ntrials))) +
  geom_point(position = position_dodge(.1)) +
  theme_bw() +
  geom_hline(yintercept=-1) +
  xlab("n_trials") +
  stat_summary(geom = "point",fun = "mean",col = "black",size = 3,shape = 24,fill = "red") +
  theme(legend.position="none")

p_mse <- ggplot(df, aes(x=as.factor(ntrials), y=mse,)) +
  geom_point(position = position_dodge(1)) +
  theme_bw() +
  xlab("n_trials") +
  theme(legend.position="none")

p_C <- ggplot(df, aes(x=C_fit, y=sigma_fit)) +
  geom_point(col="blue") +
  ylim(c(0,1)) +
  theme_bw() +
  facet_grid(~ntrials) +
  geom_hline(yintercept=df$sigma_sim) +
  theme(legend.position="none")

dev.off()
pdf("recovery_ntrials_niters_AR99.pdf",paper = "USr",width = 20,height = 20)

grid.arrange(p_sens,p_bias,p_pc,p_pe,p_sigma,p_mse, ncol = 2)
dev.off()

