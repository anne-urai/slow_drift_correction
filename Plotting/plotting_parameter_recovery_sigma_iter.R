
library(rstudioapi)
library(ggplot2)
library(readr)
library(grid)
library(gridExtra)

curdir <- dirname(getSourceEditorContext()$path)
setwd(curdir)

df <- read_csv("parameter_recovery_ntrials_niterationsAR99.csv")

p_sens <- ggplot(df, aes(x=as.factor(sens_sim), y=sens_fit, col = as.factor(n_iters))) +
  geom_point(position = position_dodge(0.1)) +
  theme_bw() +
  geom_hline(yintercept=10) +
  xlab("sens_sim") +
  #labs(col = "n_iterations", x = "sens_sim") +
  facet_grid(~ntrials) +
  theme(legend.position="none")

p_bias <- ggplot(df, aes(x=as.factor(bias_sim), y=bias_fit, col = as.factor(n_iters))) +
  geom_point(position = position_dodge(.1)) +
  theme_bw() +
  facet_grid(~ntrials) +
  geom_hline(yintercept=-5) +
  xlab("bias_sim") +
  #labs(col = "n_iterations", x = "bias_sim") +
  theme(legend.position="none")

p_sigma <- ggplot(df, aes(x=as.factor(sigma_sim), y=sigma_fit, col = as.factor(n_iters))) +
  geom_point(position = position_dodge(.1)) +
  theme_bw() +
  geom_hline(yintercept=.05) +
  xlab("sigma_sim") +
  facet_grid(~ntrials) +
  theme(legend.position="none")

p_pc <- ggplot(df, aes(x=as.factor(pc_sim), y=pc_fit, col = as.factor(n_iters))) +
  geom_point(position = position_dodge(.1)) +
  theme_bw() +
  geom_hline(yintercept=1) +
  xlab("pc_sim") +
  facet_grid(~ntrials) +
  theme(legend.position="none")

p_pe <- ggplot(df, aes(x=as.factor(pe_sim), y=pe_fit, col = as.factor(n_iters))) +
  geom_point(position = position_dodge(.1)) +
  theme_bw() +
  geom_hline(yintercept=-1) +
  xlab("pe_sim") +
  facet_grid(~ntrials) +
  theme(legend.position="none")

p_mse <- ggplot(df, aes(x=as.factor(n_iters), y=mse,)) +
  geom_point(position = position_dodge(1)) +
  theme_bw() +
  xlab("n_iterations") +
  facet_grid(~ntrials) +
  theme(legend.position="none")


dev.off()
pdf("recovery_ntrials_niters_AR99.pdf",paper = "USr",width = 20,height = 20)

grid.arrange(p_sens,p_bias,p_pc,p_pe,p_sigma,p_mse, ncol = 2)
dev.off()

