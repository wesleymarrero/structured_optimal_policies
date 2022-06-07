# ***************************************************
# Hypothesis Testing for Prices of Interpretability 
# ***************************************************

# Setup -------------------------------------------------------------------

# Setting working directory # change to appropriate directory
rm(list = ls()[!(ls() %in% c())]) #Remove all variables
home_dir = "~/Monotone Policies"

#Loading packages
if(!("data.table" %in% installed.packages()[,"Package"])) install.packages("data.table"); library(data.table) # large data frame manipulation


# Loading and Splitting Data ----------------------------------------------

# Loading Data
setwd(home_dir)
pidata = fread("Price of Interpretability per Policy.csv", stringsAsFactors = T)
colnames(pidata)[match(c("Class-Ordered Monotone","Monotone"), colnames(pidata))] = c("cmp", "mp") # renaming columns

# Dividing data accordng to BP group
bp_cat_labels = c("Normal BP", "Elevated BP", "Stage 1 Hypertension", "Stage 2 Hypertension") # BP groups' labels
pidata_bp = list()
for (i in seq(bp_cat_labels)){
  pidata_bp[[i]] = pidata[bp_cat == bp_cat_labels[i],]
}


# Testing Hypotheses ------------------------------------------------------

# Performing Wilcoxon Signed Rank Tests
## Overall
wilcox_overall = wilcox.test(pidata$mp, pidata$cmp, alternative = "greater", paired = TRUE, conf.int = TRUE, digits.rank = 4)

## Per BP Group
bonferroni3 = 1-(1-0.05)**(1/3)
wilcox_bp_cat = list()
for (i in seq(bp_cat_labels)){
  wilcox_bp_cat[[i]] = wilcox.test(pidata_bp[[i]]$mp, pidata_bp[[i]]$cmp, alternative = "greater", 
                                   paired = TRUE, conf.int = TRUE, conf.level = 1-bonferroni3, digits.rank = 4)
}
names(wilcox_bp_cat) = bp_cat_labels


# Results for paper
round(wilcox_overall$conf.int[1],4)
data.frame(ci=sapply(wilcox_bp_cat, function(x) round(x$conf.int[1],4)), p=sapply(wilcox_bp_cat, function(x) round(x$p.value,4)))
