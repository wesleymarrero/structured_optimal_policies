# *************************************************
# Monotone Policies - Continuous NHANES Extraction
# *************************************************

# Setup -------------------------------------------------------------------

rm(list = ls()[!(ls() %in% c())]) #Removing all variables

#Selecting home directory # change to appropriate directory
home_dir = getwd()#"~/Monotone Policies/Data/Continuous NHANES"
setwd(home_dir)

#Loading packages (and installing if necessary)
if(!("foreign" %in% installed.packages()[,"Package"])) install.packages("foreign"); library(foreign) #read SAS files
if(!("stringr" %in% installed.packages()[,"Package"])) install.packages("stringr"); library(stringr) #split columns
if(!("doParallel" %in% installed.packages()[,"Package"])) install.packages("doParallel"); library(doParallel) #parallel computation
if(!("missForest" %in% installed.packages()[,"Package"])) install.packages("missForest"); library(missForest) #imputation with random forest
if(!("data.table" %in% installed.packages()[,"Package"])) install.packages("data.table"); library(data.table) #data tables
if(!("biglm" %in% installed.packages()[,"Package"])) install.packages("biglm"); library(biglm) #generalized linear models for large datasets

cat("\014") #clear console

# Loading Data ------------------------------------------------------------

# Data downloaded from: https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx

#Importing demographic information
dem9 = read.xport("DEMO_F.XPT")
dem11 = read.xport("DEMO_G.XPT")
dem13 = read.xport("DEMO_H.XPT")
dem15 = read.xport("DEMO_I.XPT")

#Importing blood pressure readings
bp9 = read.xport("BPX_F.XPT")
bp11 = read.xport("BPX_G.XPT")
bp13 = read.xport("BPX_H.XPT")
bp15 = read.xport("BPX_I.XPT")

#Importing TC information
tc9 = read.xport("TCHOL_F.XPT")
tc11 = read.xport("TCHOL_G.XPT")
tc13 = read.xport("TCHOL_H.XPT")
tc15 = read.xport("TCHOL_I.XPT")

#Importing HDL information
hdl9 = read.xport("HDL_F.XPT")
hdl11 = read.xport("HDL_G.XPT")
hdl13 = read.xport("HDL_H.XPT")
hdl15 = read.xport("HDL_I.XPT")

#Importing LDL information
ldl9 = read.xport("TRIGLY_F.XPT")
ldl11 = read.xport("TRIGLY_G.XPT")
ldl13 = read.xport("TRIGLY_H.XPT")
ldl15 = read.xport("TRIGLY_I.XPT")

#Importing diabetes information
diab9 = read.xport("DIQ_F.XPT")
diab11 = read.xport("DIQ_G.XPT")
diab13 = read.xport("DIQ_H.XPT")
diab15 = read.xport("DIQ_I.XPT")

#Importing smoking information (SMQ680 - Used tobacco/nicotine last 5 days? for 2009-2012 and SMQ681 - Smoked tobacco last 5 days? for 2013-2016)
smoke9 = read.xport("SMQRTU_F.XPT")
smoke11 = read.xport("SMQRTU_G.XPT")
smoke13 = read.xport("SMQRTU_H.XPT")
smoke15 = read.xport("SMQRTU_I.XPT")

#Importing CVD information
cvd9 = read.xport("MCQ_F.XPT")
cvd11 = read.xport("MCQ_G.XPT")
cvd13 = read.xport("MCQ_H.XPT")
cvd15 = read.xport("MCQ_I.XPT")

#Joining datasets
dfs = list(dem9,bp9,tc9,hdl9,ldl9,diab9,smoke9,cvd9)
data9 = Reduce(function(...) merge(..., all.x=T,by = "SEQN"), dfs)

dfs = list(dem11,bp11,tc11,hdl11,ldl11,diab11,smoke11,cvd11)
data11 = Reduce(function(...) merge(..., all.x=T,by = "SEQN"), dfs)

dfs = list(dem13,bp13,tc13,hdl13,ldl13,diab13,smoke13,cvd13)
data13 = Reduce(function(...) merge(..., all.x=T,by = "SEQN"), dfs)

dfs = list(dem15,bp15,tc15,hdl15,ldl15,diab15,smoke15,cvd15)
data15 = Reduce(function(...) merge(..., all.x=T,by = "SEQN"), dfs)

rm(list = ls()[!(ls() %in% c("home_dir","data9","data11","data13","data15"))]); gc()

#Calculating 8-year weights
data9$WTMEC8YR = data9$WTMEC2YR*1/4 #WTMEC2YR was used instead of WTINT2YR because some variables were taken in the MEC
data11$WTMEC8YR = data11$WTMEC2YR*1/4
data13$WTMEC8YR = data13$WTMEC2YR*1/4
data15$WTMEC8YR = data15$WTMEC2YR*1/4

#Removing Unnesary Information
#change SMQ680 or SMQ681 for SMQ040 if using SMQ files intead of SMQRTU files for smoking status
data9 = data.frame("YEARS" = "2009-2010",data9[,c("SEQN","WTMEC8YR","RIDAGEYR","RIAGENDR","RIDRETH1","BPXSY1","BPXDI1","LBXTC","LBDHDD","LBDLDL","SMQ680","DIQ010","MCQ160F","MCQ160E")])
data11 = data.frame("YEARS" = "2011-2012",data11[,c("SEQN","WTMEC8YR","RIDAGEYR","RIAGENDR","RIDRETH1","BPXSY1","BPXDI1","LBXTC","LBDHDD","LBDLDL","SMQ680","DIQ010","MCQ160F","MCQ160E")])
data13 = data.frame("YEARS" = "2013-2014",data13[,c("SEQN","WTMEC8YR","RIDAGEYR","RIAGENDR","RIDRETH1","BPXSY1","BPXDI1","LBXTC","LBDHDD","LBDLDL","SMQ681","DIQ010","MCQ160F","MCQ160E")])
data15 = data.frame("YEARS" = "2015-2016",data15[,c("SEQN","WTMEC8YR","RIDAGEYR","RIAGENDR","RIDRETH1","BPXSY1","BPXDI1","LBXTC","LBDHDD","LBDLDL","SMQ681","DIQ010","MCQ160F","MCQ160E")])
colnames(data13) = colnames(data15) = colnames(data11)

#Combining datasets
ndata = rbind(data9,data11,data13,data15)
rm(list = ls()[!(ls() %in% c("home_dir","ndata"))]); gc()

save(ndata,file = "Raw_Continuous_NHANES_Dataset.RData")

# Modifiying Dataset ------------------------------------------------------

#Loading data
load("Raw_Continuous_NHANES_Dataset.RData")

#Renaming columns
cnames = read.csv("Continuous_NHANES_Variables.csv")
colnames(ndata) = as.character(cnames$Meaning[match(names(ndata),as.character(cnames$Variable))]) #renaming columns

#Counting records
nrow(ndata)
sum(ndata$WEIGHT)/1e06

#Restricting Analysis for ages 40 to 75
length(ndata$WEIGHT[ndata$AGE<40 | ndata$AGE>75]) #age exclusions
sum(ndata$WEIGHT[ndata$AGE<40 | ndata$AGE>75])/1e06 #age exclusions
ndata = ndata[ndata$AGE>=40 & ndata$AGE<=75,]

ndata$YEARS = factor(ndata$YEARS) #Converting years into categorical variable

ndata$SEX = factor(ndata$SEX) #Converting sex into categorical variable
levels(ndata$SEX)=list("Male"=1,"Female"=2)

ndata$RACE = factor(ndata$RACE) #Converting race into categorical variable
levels(ndata$RACE)=list("White"=3,"Black"=4,"Hispanic"=1,"Hispanic"=2,"Other"=5)

ndata$SMOKER = factor(ndata$SMOKER) #Converting smoking into categorical variable
levels(ndata$SMOKER)=list("N"=2,"Y"=1,"U"=7,"U"=9)#;relevel(ndata$SMOKER,ref = "N")

ndata$DIABETIC = factor(ndata$DIABETIC) #Converting diabetic into categorical variable
levels(ndata$DIABETIC)=list("N"=2,"N"=3,"Y"=1,"U"=7,"U"=9)#;relevel(ndata$DIABETIC,ref = "N")

ndata$STROKE = factor(ndata$STROKE) #Converting stroke into categorical variable
levels(ndata$STROKE)=list("N"=2,"Y"=1,"U"=7,"U"=9)#;relevel(ndata$STROKE,ref = "N")

ndata$CHD = factor(ndata$CHD) #Converting CHD into categorical variable
levels(ndata$CHD)=list("N"=2,"Y"=1,"U"=7,"U"=9)#;relevel(ndata$CHD,ref = "N")

#Removing Unknowns
ndata[ndata == "U"] = NA

#Removing 0's from DBP
ndata$DBP = ifelse(ndata$DBP==0,NA,ndata$DBP)

#Removing DBPs with numbers too low
ndata$DBP = ifelse(ndata$DBP<10,NA,ndata$DBP)

#Excluding patients with history of CHD or stroke and removing variables
length(ndata$WEIGHT[which(ndata$CHD=="Y"|ndata$STROKE=="Y")]) #history of CVD exclusions
sum(ndata$WEIGHT[which(ndata$CHD=="Y"|ndata$STROKE=="Y")]) #history of CVD exclusions
ndata = ndata[-which(ndata$CHD=="Y"|ndata$STROKE=="Y"),]
cols = match(c("CHD","STROKE"),names(ndata))
ndata = ndata[,-cols]

#Excluding Hipanic and Other race patients (not in ASCVD risk)
length(ndata$WEIGHT[which(ndata$RACE=="Hispanic"|ndata$RACE=="Other")]) #race exclusions
sum(ndata$WEIGHT[which(ndata$RACE=="Hispanic"|ndata$RACE=="Other")]) #race exclusions
ndata = ndata[-which(ndata$RACE=="Hispanic"|ndata$RACE=="Other"),]

#Counting dataset after exclusions
ndata = droplevels(ndata)
length(ndata$WEIGHT)
sum(ndata$WEIGHT)/1e06
save(ndata,file = "Continuous_NHANES_Dataset.RData")

# Performing Multiple Imputation ------------------------------------------

#Loading preprocessed dataset
load("Continuous_NHANES_Dataset.RData")

#Parallel computing parameters
cl = makeCluster(detectCores()-1)
registerDoParallel(cl)

#Imputing data using random forest
impdata = missForest(ndata, parallelize = "forest")
impdata = impdata$ximp

stopCluster(cl)

#Adding age squared column and reorganizing columns
impdata$AGE2 = impdata$AGE^2
cols = match(c("YEARS","SEQN","WEIGHT","AGE","AGE2","SEX","RACE","SMOKER","DIABETIC","SBP","DBP","TC","HDL","LDL"),names(impdata))
impdata = impdata[,cols]

save(impdata,file = "Continuous NHANES Imputed Dataset.RData")

# Linear Models for Continuous Variables ----------------------------------

#Loading preprocessed dataset
setwd(home_dir)
load("Continuous NHANES Imputed Dataset.RData")
setDT(impdata)

#Converting categorical variable into factors
fcols = c("YEARS","SEX","RACE","SMOKER","DIABETIC")
impdata[,(fcols):=lapply(.SD, as.factor),.SDcols=fcols]

#Setting reference levels
impdata$SEX = relevel(impdata$SEX,ref = "Male")
impdata$RACE = relevel(impdata$RACE,ref = "White")

#Defining columns of predictors
cols = match(c("AGE","AGE2","SEX","RACE","SMOKER","DIABETIC"),names(impdata))

#Linear regression for SBP
fm = as.formula(paste("SBP~",paste(colnames(impdata)[cols],collapse = "+")))
full = lm(fm,data=impdata) #Full model
svs = step(full,direction="both",trace = 0) #Stepwise variable selection
sbp = biglm(formula(svs),data=impdata) #memory efficient lm for saving

print(paste("sbp done ",Sys.time(),sep = ""))

#Linear regression for DBP
fm = as.formula(paste("DBP~",paste(colnames(impdata)[cols],collapse = "+")))
full = lm(fm,data=impdata) #Full model
svs = step(full,direction="both",trace = 0) #Stepwise variable selection
dbp = biglm(formula(svs),data=impdata) #memory efficient lm for saving

print(paste("dbp done ",Sys.time(),sep = ""))

#Linear regression for TC
fm = as.formula(paste("TC~",paste(colnames(impdata)[cols],collapse = "+")))
full = lm(fm,data=impdata) #Full model
svs = step(full,direction="both",trace = 0) #Stepwise variable selection
tc = biglm(formula(svs),data=impdata) #memory efficient lm for saving

print(paste("tc done ",Sys.time(),sep = ""))

#Linear regression for HDL
fm = as.formula(paste("HDL~",paste(colnames(impdata)[cols],collapse = "+")))
full = lm(fm,data=impdata) #Full model
svs = step(full,direction="both",trace = 0) #Stepwise variable selection
hdl = biglm(formula(svs),data=impdata) #memory efficient lm for saving

print(paste("hdl done ",Sys.time(),sep = ""))

#Linear regression for LDL
fm = as.formula(paste("LDL~",paste(colnames(impdata)[cols],collapse = "+")))
full = lm(fm,data=impdata) #Full model
svs = step(full,direction="both",trace = 0) #Stepwise variable selection
ldl = biglm(formula(svs),data=impdata) #memory efficient lm for saving

print(paste("ldl done ",Sys.time(),sep = ""))

#Deleting unnecesary files and cleaning memory
rm(full,svs);gc()

save(sbp,dbp,tc,hdl,ldl,file = "Linear regression models.RData")

# Forecasting Continuous Variables ----------------------------------------

print(paste("Performing forecast ",Sys.time(),sep = ""))

#Loading data
setwd(home_dir)
load("Continuous NHANES Imputed Dataset.RData")
impdata = impdata[impdata$AGE>=40 & impdata$AGE<=60,] # using ages 40 to 60 for analysis in Python (without weight expansion to save computational burden)
setDT(impdata)
load("Linear regression models.RData")

#Converting categorical variable into factors
fcols = c("SEX","RACE","SMOKER","DIABETIC")
impdata[,(fcols):=lapply(.SD, as.factor),.SDcols=fcols]

#Setting no treatment as reference level in drug treatment
impdata$SEX = relevel(impdata$SEX,ref = "Male")
impdata$RACE = relevel(impdata$RACE,ref = "White")

print(paste("Loading done ",Sys.time(),sep = ""))

#Expanding dataset for ten year forecast
patientdata = impdata[rep(seq(.N), each=10)]

print(paste("Expansion done ",Sys.time(),sep = ""))

#Saving age from impdata
age = impdata[,AGE]

#Caculating intercept adjustements
intercept_sbp = rep(c(impdata[,SBP]-predict(sbp,newdata = impdata)),each=10)
intercept_dbp = rep(c(impdata[,DBP]-predict(dbp,newdata = impdata)),each=10)
intercept_tc = rep(c(impdata[,TC]-predict(tc,newdata = impdata)),each=10)
intercept_hdl = rep(c(impdata[,HDL]-predict(hdl,newdata = impdata)),each=10)
intercept_ldl = rep(c(impdata[,LDL]-predict(ldl,newdata = impdata)),each=10)

#Removing impdata from memory
rm(impdata);gc()

#Filling age
patientdata[,AGE := (rep(age,each = 10)+0:9)]

#Calculating squared age
patientdata[,AGE2 := AGE^2]

print(paste("Age done ",Sys.time(),sep = ""))

#Linear regression for SBP
patientdata[,SBP := predict(sbp,newdata = patientdata)+(intercept_sbp)]
print(paste("sbp done ",Sys.time(),sep = ""))

#Linear regression for DBP
patientdata[,DBP := predict(dbp,newdata = patientdata)+(intercept_dbp)]
print(paste("dbp done ",Sys.time(),sep = ""))

#Linear regression for TC
patientdata[,TC := predict(tc,newdata = patientdata)+(intercept_tc)]
print(paste("tc done ",Sys.time(),sep = ""))

#Linear regression for HDL
patientdata[,HDL := predict(hdl,newdata = patientdata)+(intercept_hdl)]
print(paste("hdl done ",Sys.time(),sep = ""))

#Linear regression for LDL
patientdata[,LDL := predict(ldl,newdata = patientdata)+(intercept_ldl)]
print(paste("ldl done ",Sys.time(),sep = ""))

#Chaging SEQN number for ordered ID
patientdata[,SEQN := rep(0:((nrow(patientdata)-1)/10), each=10)]

#Removing unnecessary variables
patientdata[, c("YEARS","AGE2"):=NULL]

#Formatting data for Computations
patientdata[, SEX := ifelse(SEX=="Male",1,0)]
patientdata[, RACE := ifelse(RACE=="White",1,0)]
patientdata[, SMOKER := ifelse(SMOKER=="Y",1,0)]
patientdata[, DIABETIC := ifelse(DIABETIC=="Y",1,0)]
colnames(patientdata) = c("id","wt","age","sex","race","smk","diab","sbp","dbp","tc","hdl","ldl") #Column names for Continuous NHANES

# Uusing sampling weights as recorded in NHANES to reduce computational burden
fwrite(patientdata,"Continuous NHANES Forecasted Dataset.csv")

