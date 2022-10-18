
library(rstudioapi)
library(ggplot2)
library(readr)
library(grid)
library(gridExtra)

curdir <- dirname(getSourceEditorContext()$path)
setwd(curdir)

df <- read_csv("param_recovery_originalGupta.csv")

p_sens <- ggplot(df, aes(x=as.factor(ntrials), y=sens_fit, col = as.factor(ntrials))) +
  geom_hline(yintercept=df$sens_sim) +
  geom_point(position = position_dodge(0.1)) +
  theme_bw() +
  ylim(c(6,14)) +
  stat_summary(geom = "point",fun = "mean",col = "black",size = 3,shape = 24,fill = "red") +
  labs(x = "Number of trials", y = "Perceptual sensitivity") +
  theme(legend.position="none")

p_bias <- ggplot(df, aes(x=as.factor(ntrials), y=bias_fit, col = as.factor(ntrials))) +
  geom_hline(yintercept=df$bias_sim) +
  geom_point(position = position_dodge(.1)) +
  theme_bw() +
  ylim(c(-8,-2)) +
  stat_summary(geom = "point",fun = "mean",col = "black",size = 3,shape = 24,fill = "red") +
  labs(x = "Number of trials", y = "Intercept") +
  theme(legend.position="none")

p_sigma <- ggplot(df, aes(x=as.factor(ntrials), y=sigma_fit, col = as.factor(ntrials))) +
  geom_hline(yintercept=df$sigma_sim) +
  geom_point(position = position_dodge(.1)) +
  theme_bw() +
  #ylim(c(0,.8)) +
  labs(x = "Number of trials", y = "Sigma") +
  stat_summary(geom = "point",fun = "mean",col = "black",size = 3,shape = 24,fill = "red") +
  theme(legend.position="none")

p_pc <- ggplot(df, aes(x=as.factor(ntrials), y=pc_fit, col = as.factor(ntrials))) +
  geom_hline(yintercept=df$pc_sim) +
  geom_point(position = position_dodge(.1)) +
  theme_bw() +
  ylim(c(.4,1.6)) +
  labs(x = "Number of trials", y = "Post correct (PC)") +
  stat_summary(geom = "point",fun = "mean",col = "black",size = 3,shape = 24,fill = "red") +
  theme(legend.position="none")

mean(df$pe_fit[df$ntrials==10000])
p_pe <- ggplot(df, aes(x=as.factor(ntrials), y=pe_fit, col = as.factor(ntrials))) +
  geom_hline(yintercept=df$pe_sim) +
  geom_point(position = position_dodge(.1)) +
  theme_bw() +
  ylim(c(-1.6,-.4)) +
  labs(x = "Number of trials", y = "Post error (PE)") +
  stat_summary(geom = "point",fun = "mean",col = "black",size = 3,shape = 24,fill = "red") +
  theme(legend.position="none")


p_C <- ggplot(df, aes(x=C_fit, y=sigma_fit)) +
  geom_hline(yintercept=df$sigma_sim) +
  geom_point(col="blue") +
  #ylim(c(0,.75)) +
  #scale_x_continuous(breaks=c(-1,0,1)) + 
  theme_bw() +
  facet_grid(~ntrials) +
  labs(x = "C", y = "Sigma") +
  theme(legend.position="none")

dev.off()
pdf("param_recovery_ntrials_evieffectcoded_driftnotmeancorrected.pdf",paper = "USr",width = 20,height = 20)
grid.arrange(p_sens,p_bias,p_pc,p_pe,p_sigma,p_C, ncol = 2)
dev.off()


