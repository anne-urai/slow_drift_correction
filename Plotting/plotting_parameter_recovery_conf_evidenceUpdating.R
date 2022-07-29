
library(rstudioapi)
library(ggplot2)
library(readr)
library(grid)
library(gridExtra)

curdir <- dirname(getSourceEditorContext()$path)
setwd(curdir)

df <- read_csv("parameter_recoveryConfEvidence_1111.csv")

p_sens <- ggplot(df, aes(x=as.factor(sens_sim), y=sens_fit)) +
  geom_point(col = "blue") +
  theme_bw() +
  ylim(c(5,15)) +
  geom_hline(yintercept=df$sens_sim) +
  xlab("sens_sim") +
  theme(legend.position="none")

p_bias <- ggplot(df, aes(x=as.factor(bias_sim), y=bias_fit)) +
  geom_point(col = "blue") +
  theme_bw() +
  ylim(c(-10,0)) +
  geom_hline(yintercept=df$bias_sim) +
  xlab("bias_sim") +
  theme(legend.position="none")

p_sigma <- ggplot(df, aes(x=as.factor(sigma_sim), y=sigma_fit)) +
  geom_point(col = "blue") +
  theme_bw() +
  ylim(c(0,1)) +
  geom_hline(yintercept=df$sigma_sim) +
  xlab("sigma_sim") +
  theme(legend.position="none")

p_C <- ggplot(df, aes(x=c_fit, y=sigma_fit)) +
  geom_point(col = "blue") +
  ylim(c(0,1)) +
  theme_bw() +
  geom_hline(yintercept=df$sigma_sim) +
  theme(legend.position="none")

p_resp <- ggplot(df, aes(x=as.factor(prevresp_sim), y=prevresp_fit)) +
  geom_point(col = "blue") +
  theme_bw() +
  ylim(c(-2,2)) +
  geom_hline(yintercept=df$prevresp_sim) +
  xlab("prevresp_sim") +
  theme(legend.position="none")

p_conf <- ggplot(df, aes(x=as.factor(prevconf_sim), y=prevconf_fit)) +
  geom_point(col = "blue") +
  theme_bw() +
  ylim(c(-2,2)) +
  geom_hline(yintercept=df$prevconf_sim) +
  xlab("prevconf_sim") +
  theme(legend.position="none")

p_respconf <- ggplot(df, aes(x=as.factor(prevconfprevresp_sim), y=prevconfprevresp_fit)) +
  geom_point(col = "blue") +
  theme_bw() +
  ylim(c(-3,3)) +
  geom_hline(yintercept=df$prevconfprevresp_sim) +
  xlab("prevconfprevresp_sim") +
  theme(legend.position="none")

p_sign <- ggplot(df, aes(x=as.factor(prevsignevi_sim), y=prevsignevi_fit)) +
  geom_point(col = "blue") +
  theme_bw() +
  ylim(c(-3,3)) +
  geom_hline(yintercept=df$prevsignevi_sim) +
  xlab("prevsignevi_sim") +
  theme(legend.position="none")

p_absevi <- ggplot(df, aes(x=as.factor(prevabsevi_sim), y=prevabsevi_fit)) +
  geom_point(col = "blue") +
  theme_bw() +
  ylim(c(-2,2)) +
  geom_hline(yintercept=df$prevabsevi_sim) +
  xlab("prevabsevi_sim") +
  theme(legend.position="none")

p_signabsevi <- ggplot(df, aes(x=as.factor(prevsignevi_prevabsevi_sim), y=prevsignevi_prevabsevi_fit)) +
  geom_point(col = "blue") +
  theme_bw() +
  ylim(c(-3,3)) +
  geom_hline(yintercept=df$prevsignevi_prevabsevi_sim) +
  xlab("prevsignevi_prevabsevi_sim") +
  theme(legend.position="none")

p_mse <- ggplot(df, aes(x=as.factor(n_iters), y=mse,)) +
  geom_point(col = "blue",position = position_dodge(1)) +
  theme_bw() +
  ylim(c(0,10)) +
  xlab("n_iterations") +
  theme(legend.position="none")


dev.off()
pdf("1001_largerSigma.pdf",paper = "USr",width = 20,height = 20)

grid.arrange(p_sens,p_bias,p_resp,p_conf,p_respconf,p_sign,p_absevi,p_signabsevi,p_sigma,p_C,p_mse, ncol = 3)
dev.off()

