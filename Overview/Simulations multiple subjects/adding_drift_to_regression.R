
library(rstudioapi)
library(lme4)
library(car)
library(effects)


curdir <- dirname(getSourceEditorContext()$path)
setwd(curdir)

# drift coef .98, sigma .4, no sys updating -> over 90 subjects
# adding drift to mixed model doesn't make the apparent systematic updating go away!
# whereas this still works for .9995, sigma .1
input <- read.csv("df_input.csv", header = TRUE)
choices <- read.csv("df_choices.csv", header = TRUE)
drift <- read.csv("df_drift.csv", header = TRUE)
estDrift <- read.csv("df_estdrift.csv", header = TRUE)

sub <- rep(1:90,each=500)
df <- cbind(input,choices$X0,drift$X0,estDrift$X0,sub)

names(df) <- c("trial","evidence","prevresp","prevconf","prevrespconf","prevsignevi","prevabsevi","prevsignabsevi","resp","drift","estdrift","sub")
df <- df[!(df$prevresp == 0),] # first trial of each subject is nonsens so remove

df$prevresp <- as.factor(df$prevresp)
df$prevsignevi <- as.factor(df$prevsignevi)

# drift with or without intercept added doesn't matter, it's just a constant being added
# could play a role though when merging subjects with each their own intercept
m <- glm(data=df, resp ~ 
            
            evidence +
            prevresp * prevconf + 
            prevsignevi * prevabsevi + estdrift,
          family = binomial)
vif(m)
summary(m)
Anova(m)
plot(effect(c('prevresp'), m))
plot(effect(c('prevresp:prevconf'), m))


mm <- glmer(data=df, resp ~ 
                   
              evidence +
              prevresp * prevconf +
              prevsignevi * prevabsevi + drift +
                   
                (1|sub),
                 family = binomial,control=glmerControl(optimizer='bobyqa',optCtrl = list(maxfun=100000)))

vif(mm)
summary(mm)
Anova(mm)
plot(effect(c('prevresp'), mm))
plot(effect(c('prevresp:prevconf'), mm))
