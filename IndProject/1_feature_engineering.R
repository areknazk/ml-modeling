#### Set-Up ####

# source relevant files
source('load_libraries.R')

# read in data 
raw_data_train<-fread('data/BlackFriday_train.csv', stringsAsFactors = F)
raw_data_test<-fread('data/BlackFriday_test.csv', stringsAsFactors=F)

# Create "empty" Purchase column for raw_data_test so it can be merged with raw_data_train for pre-processing 
raw_data_test$Purchase<-0
raw_data <- rbind(raw_data_train, raw_data_test)

str(raw_data)
summary(raw_data)

#### Introduction ####

# Objective: predict the spending of a person on Black Friday

# We have character,integer, and numeric variables
# In reality all the variables are meant to be characters except Purchase (which is correctly identified as num)

raw_data[, names(raw_data)[names(raw_data)!="Purchase"]:=lapply(.SD,as.character),
         .SDcols = names(raw_data)[names(raw_data)!="Purchase"]]

str(raw_data)

# lets now turn characters into factors

data_proc <- raw_data
data_proc[ , names(data_proc)[sapply(data_proc, is.character)]:=lapply(.SD,as.factor),
           .SDcols = names(data_proc)[sapply(data_proc, is.character)]]
str(data_proc)

# We drop off a couple of variables with non interest for our goal
data_proc[, c('User_ID','Product_ID'):=NULL]

str(data_proc)


#### summary
str(data_proc)  # ...just numeric & factor variables

sum(sapply(data_proc, is.numeric))
sum(sapply(data_proc, is.factor))



#### NA treatment ####

sum(is.na(data_proc))


# We first delete every row where Purchase is missing
nrow(data_proc)
data_proc<-data_proc[!is.na(Purchase)]
nrow(data_proc)

# plotting our problem
library(Amelia)
suppressWarnings(
  missmap(data_proc, legend = F, col=c('black', 'lightgray'))
); grid(col='azure4')

sapply(data_proc, function(x) sum(is.na(x)))

# NAs are just in Product_Category_2/3, we can fill them in with Product_Category_1/2 values
data_proc[is.na(data_proc$Product_Category_2),"Product_Category_2"]<-
  data_proc[is.na(data_proc$Product_Category_2),"Product_Category_1"]

data_proc[is.na(data_proc$Product_Category_3),"Product_Category_3"]<-
  data_proc[is.na(data_proc$Product_Category_3),"Product_Category_2"]

# We can also try just filling the NA's with "0" to have less duplication of Product_Category_1
# (this was skipped, because in training it did not improve model performance)
#data_proc[is.na(data_proc$Product_Category_2),"Product_Category_2"]<-"0"
#data_proc[is.na(data_proc$Product_Category_3),"Product_Category_3"]<-"0"

sapply(data_proc, function(x) sum(is.na(x)))

#### Other pre-processing ####
#### We check if any numeric variable has null variance

numeric_variables<-names(data_proc)[sapply(data_proc, is.numeric)]

# calculating sd and CV for every numeric variable
sd_numeric_variables<-sapply(data_proc[,numeric_variables, with=F], sd)
sd_numeric_variables
cv_numeric_variables<-sd_numeric_variables/colMeans(data_proc[,numeric_variables, with=F])
cv_numeric_variables

ggplot(data.table(var=names(cv_numeric_variables),cv=cv_numeric_variables),
       aes(var,fill=cv))+geom_bar()+coord_polar()+scale_fill_gradient(low='white', high = 'black')

# Now lets check the number of categories per factor variable
factor_variables<-names(data_proc)[sapply(data_proc, is.factor)]
count_factor_variables<-sapply(data_proc[,factor_variables, with=F], summary)
count_factor_variables

# lets define a baseline rule... if a label weight less than 5% goes into the "others" bag:
f_other<-function(var){
  
  count_levels<-summary(var)/length(var)
  to_bag<-names(which(count_levels<0.05))
  
  reduced_var<-as.factor(ifelse(as.character(var)%in%to_bag,'others',as.character(var)))
  
  return(reduced_var)
}


# now let's see if any of the "others" should be more clearly divided into high/low Purchase amounts
P_C1_others<-data_proc[data_proc$Purchase>0,c(list(perc = .N/483819),list(avg_purch=median(Purchase))),by=Product_Category_1][order(perc)][perc<0.05]
P_C2_others<-data_proc[data_proc$Purchase>0,c(list(perc = .N/483819),list(avg_purch=median(Purchase))),by=Product_Category_2][order(perc)][perc<0.05]
P_C3_others<-data_proc[data_proc$Purchase>0,c(list(perc = .N/483819),list(avg_purch=median(Purchase))),by=Product_Category_3][order(perc)][perc<0.05]
Occ_others<-data_proc[data_proc$Purchase>0,c(list(perc = .N/483819),list(avg_purch=median(Purchase))),by=Occupation][order(perc)][perc<0.05]

med_Purchase<-data_proc[data_proc$Purchase>0,median(Purchase)]

P_C1_others$high_low[P_C1_others$avg_purch>med_Purchase]<-"high"
P_C1_others$high_low[P_C1_others$avg_purch<=med_Purchase]<-"low"

P_C2_others$high_low[P_C2_others$avg_purch>med_Purchase]<-"high"
P_C2_others$high_low[P_C2_others$avg_purch<=med_Purchase]<-"low"

P_C3_others$high_low[P_C3_others$avg_purch>med_Purchase]<-"high"
P_C3_others$high_low[P_C3_others$avg_purch<=med_Purchase]<-"low"

Occ_others$high_low[Occ_others$avg_purch>med_Purchase]<-"high"
Occ_others$high_low[Occ_others$avg_purch<=med_Purchase]<-"low"

# apply the high/low categories to their respective factor variables using the following function
f_high_low<-function(var, others_var, others_high_low){
  
  var <- as.character(var)
  var[var%in%others_var[others_high_low=="high"]]<-"high"
  var[var%in%others_var[others_high_low=="low"]]<-"low"
  
  reduced_var<-as.factor(var)
  
  return(reduced_var)
}

data_proc$Product_Category_1<-
  f_high_low(data_proc$Product_Category_1, P_C1_others$Product_Category_2, P_C1_others$high_low)

data_proc$Product_Category_2<-
  f_high_low(data_proc$Product_Category_2, P_C2_others$Product_Category_2, P_C2_others$high_low)

data_proc$Product_Category_3<-
  f_high_low(data_proc$Product_Category_3, P_C3_others$Product_Category_3, P_C3_others$high_low)

data_proc$Occupation<-
  f_high_low(data_proc$Occupation, Occ_others$Occupation, Occ_others$high_low)

# and we apply the others function to our remaining factor variables if applicable
data_proc[, (factor_variables):=lapply(.SD, f_other), .SDcols=factor_variables]

sapply(data_proc[,factor_variables, with=F], summary)

str(data_proc)


# checking skewness in training Purchase values
hist(data_proc$Purchase[data_proc$Purchase>0])
skew<-e1071::skewness(data_proc$Purchase[data_proc$Purchase>0])

# skewness was low, so we don't need to rescale the variable


#### Binary encoding  factor variables ####

data_ready<-caret::dummyVars(formula= ~., data = data_proc, fullRank=T,sep = "_")
data_ready<-data.table(predict(data_ready, newdata = data_proc))

names(data_ready)<-gsub('-','_',names(data_ready))

str(data_proc)
str(data_ready)

# resplit the data into the train and test sets
data_ready_train <- data_ready[1:483819,]
data_ready_test <- data_ready[483820:537577,]

# save the processed files
fwrite(data_ready_train, 'data/BlackFriday_train_ready1_1.csv', row.names = F)
fwrite(data_ready_test, 'data/BlackFriday_test_ready1_1.csv', row.names = F)

