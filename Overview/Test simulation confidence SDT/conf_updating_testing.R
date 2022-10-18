library(ggplot2)
library(dplyr)
library(car)


df <- read.csv("C:/Users/u0141056/OneDrive - KU Leuven/PhD/PROJECTS/CHOICE HISTORY BIAS/Correction slow drifts/slow_drift_correction/checkconf.csv")

names(df) <- c("trial","evidence","prevresp","prevconf","prevrespconf","prevsignevi","prevabsevi","prevsignabsevi","resp","stim_side")

df$correct <- ifelse(df$resp == df$stim_side, 1,0) #correct response
df$conf <- lead(df$prevconf) #confidence

df <- df[2:(nrow(df)-1),] #drop first trial (cause 0 for prevresp) and last trial (NA for conf)

# check that mean evidence = generative means (-.5 and .5 if dprime = 1)
aggregate(df$evidence,by=list(df$stim_side),mean)

# hist conf
hist(df$conf)

# check that confidence scales with evidence
df$evidence_round <- round(df$evidence,1)
agr_evi <- aggregate(df$conf,by=list(df$evidence_round),mean)

ggplot(agr_evi, aes(x=Group.1, y=x)) +
  geom_point() +
  labs(x = "evidence", y = "confidence") +
  theme_bw()


# check that accuracy scales with confidence
df$conf_round <- round(df$conf,1)
agr_conf <- aggregate(df$correct,by=list(df$conf_round),mean)

ggplot(agr_conf, aes(x=Group.1, y=x)) +
  geom_point() +
  labs(x = "confidence", y = "accuracy") +
  theme_bw()



df$prevresp <- as.factor(df$prevresp)
fit <- glm(resp~evidence + prevresp*prevconf + prevsignevi*prevabsevi,df,family=binomial)

vif(fit)
summary(fit)
Anova(fit)

plot(effects::effect(c('prevresp:prevconf'),fit))
