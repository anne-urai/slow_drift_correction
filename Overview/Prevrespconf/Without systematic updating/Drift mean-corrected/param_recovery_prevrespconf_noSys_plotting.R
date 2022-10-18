
library(rstudioapi)
library(ggplot2)
library(readr)
library(grid)
library(gridExtra)

curdir <- dirname(getSourceEditorContext()$path)
setwd(curdir)


df <- read_csv("param_recovery_prevrespconf_noSys.csv")


p_sens <- ggplot(df, aes(x=as.factor(ntrials), y=sens_fit, col = as.factor(ntrials))) +
  geom_hline(yintercept=df$sens_sim) +
  geom_point(position = position_dodge(0.1)) +
  theme_bw() +
  ylim(c(5,15)) +
  stat_summary(geom = "point",fun = "mean",col = "black",size = 3,shape = 24,fill = "red") +
  labs(x = "Number of trials", y = "Perceptual sensitivity") +
  theme(legend.position="none")

p_bias <- ggplot(df, aes(x=as.factor(ntrials), y=bias_fit, col = as.factor(ntrials))) +
  geom_hline(yintercept=df$bias_sim) +
  geom_point(position = position_dodge(.1)) +
  theme_bw() +
  ylim(c(-5,5)) +
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


p_resp <- ggplot(df, aes(x=as.factor(ntrials), y=prevresp_fit, col = as.factor(ntrials))) +
  geom_hline(yintercept=df$prevresp_sim) +
  geom_point(position = position_dodge(0.1)) +
  theme_bw() +
  ylim(c(-1,1)) +
  stat_summary(geom = "point",fun = "mean",col = "black",size = 3,shape = 24,fill = "red") +
  labs(x = "Number of trials", y = "Prev. resp") +
  theme(legend.position="none")
mean(df$prevresp_fit[df$ntrials==50000])

p_conf <- ggplot(df, aes(x=as.factor(ntrials), y=prevconf_fit, col = as.factor(ntrials))) +
  geom_hline(yintercept=df$prevconf_sim) +
  geom_point(position = position_dodge(0.1)) +
  theme_bw() +
  ylim(c(-1,1)) +
  stat_summary(geom = "point",fun = "mean",col = "black",size = 3,shape = 24,fill = "red") +
  labs(x = "Number of trials", y = "Prev. conf") +
  theme(legend.position="none")
mean(df$prevconf_fit[df$ntrials==10000])


p_respconf <- ggplot(df, aes(x=as.factor(ntrials), y=prevrespconf_fit, col = as.factor(ntrials))) +
  geom_hline(yintercept=df$prevrespconf_sim) +
  geom_point(position = position_dodge(0.1)) +
  theme_bw() +
  ylim(c(-1,1)) +
  stat_summary(geom = "point",fun = "mean",col = "black",size = 3,shape = 24,fill = "red") +
  labs(x = "Number of trials", y = "Prev. resp * prev. conf") +
  theme(legend.position="none")
mean(df$prevrespconf_fit[df$ntrials==50000])


p_sign <- ggplot(df, aes(x=as.factor(ntrials), y=prevsignevi_fit, col = as.factor(ntrials))) +
  geom_hline(yintercept=df$prevsignevi_sim) +
  geom_point(position = position_dodge(0.1)) +
  theme_bw() +
  ylim(c(-1,1)) +
  stat_summary(geom = "point",fun = "mean",col = "black",size = 3,shape = 24,fill = "red") +
  labs(x = "Number of trials", y = "Prev. sign evi") +
  theme(legend.position="none")

p_absevi <- ggplot(df, aes(x=as.factor(ntrials), y=prevabsevi_fit, col = as.factor(ntrials))) +
  geom_hline(yintercept=df$prevabsevi_sim) +
  geom_point(position = position_dodge(0.1)) +
  theme_bw() +
  ylim(c(-1,1)) +
  stat_summary(geom = "point",fun = "mean",col = "black",size = 3,shape = 24,fill = "red") +
  labs(x = "Number of trials", y = "Prev. abs evi") +
  theme(legend.position="none")

p_signabsevi <- ggplot(df, aes(x=as.factor(ntrials), y=prevsignabsevi_fit, col = as.factor(ntrials))) +
  geom_hline(yintercept=df$prevsignabsevi_sim) +
  geom_point(position = position_dodge(0.1)) +
  theme_bw() +
  ylim(c(-1,1)) +
  stat_summary(geom = "point",fun = "mean",col = "black",size = 3,shape = 24,fill = "red") +
  labs(x = "Number of trials", y = "Prev. sign * prev. abs evi") +
  theme(legend.position="none")
mean(df$prevsignabsevi_fit[df$ntrials==10000])

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
grid.arrange(p_sens,p_bias,p_resp,p_conf,p_respconf,p_sign,p_absevi,p_signabsevi,p_sigma, ncol = 3)
dev.off()



df_cor <- df[df$ntrials==10000,]
cor(df_cor$sens_fit,df_cor$prevresp_fit)
cor(df_cor$prevsignabsevi_fit,df_cor$prevresp_fit)
