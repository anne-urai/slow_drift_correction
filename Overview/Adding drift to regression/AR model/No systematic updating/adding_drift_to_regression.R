
library(rstudioapi)
library(lme4)
library(car)
library(effects)


curdir <- dirname(getSourceEditorContext()$path)
setwd(curdir)

# simulate data with only effect of prev resp = 1

input <- read.csv("df_input.csv", header = TRUE)
choices <- read.csv("df_choices.csv", header = TRUE)
drift <- read.csv("df_drift.csv", header = TRUE)
estDrift <- read.csv("df_estdrift.csv", header = TRUE)


df <- cbind(input,choices$X0,drift$X0,estDrift$X0)
df <- df[-1,] # first row is 0 for input
names(df) <- c("trial","evidence","prevresp","prevconf","prevrespconf","prevsignevi","prevabsevi","prevsignabsevi","resp","drift","estdrift")

df$prevresp <- as.factor(df$prevresp)
df$prevsignevi <- as.factor(df$prevsignevi)

# drift with or without intercept added doesn't matter, it's just a constant being added
# could play a role though when merging subjects with each their own intercept
m <- glm(data=df, resp ~ 
            
            evidence +
            prevresp * prevconf +
            prevsignevi * prevabsevi +
           drift,
          family = binomial)
vif(m)
summary(m)
Anova(m)
plot(effect(c('prevresp'), m))
plot(effect(c('prevresp:prevconf'), m))
