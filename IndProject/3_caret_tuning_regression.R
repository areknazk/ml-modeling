# Hyperparameter tuning with caret

#### Set-Up ####
# source scripts
source('load_libraries.R')
source('f_partition.R')
source('regression_metrics.R')


# read in data from version1 feature engineering, and split into train and test sets
whole_data<-f_partition(df=fread('data/BlackFriday_train_ready1.csv'),
                        test_proportion = 0.2,
                        seed = 872367823)
str(whole_data)

# covert any integer columns to numeric
whole_data<-lapply(whole_data, function(x){
  return(x[, which(sapply(x, is.integer)):=lapply(.SD, as.numeric), .SDcols=sapply(x,is.integer)])
})
str(whole_data)

# check for non-contant variables
sum(sapply(whole_data[['train']],function(x) var(x)==0))
sum(sapply(whole_data[['test']],function(x) var(x)==0))

str(whole_data)


# load library
library(caret)

# set formula
formula<-as.formula(Purchase~.) 

# fix issue with "+"
whole_data$train$Stay_In_Current_City_Years_4<-whole_data$train$`Stay_In_Current_City_Years_4+`
whole_data$train[, c("Stay_In_Current_City_Years_4+"):=NULL]
whole_data$test$Stay_In_Current_City_Years_4<-whole_data$test$`Stay_In_Current_City_Years_4+`
whole_data$test[, c("Stay_In_Current_City_Years_4+"):=NULL]


#### ctree ####

# 1. Define grid of hyperparameters to tune
tuneGrid=data.table(expand.grid(maxdepth=c(5,10,25,50),
                                maxdepth=50,
                                mincriterion = c(0.80,0.85,0.90,0.95)))

# check dimension and grid
dim(tuneGrid)
tuneGrid


# 2. Define the validation squema
ctrl <- trainControl(
  method = "cv",
  number = 3,
  savePredictions=TRUE
)

# 3. Train the model
ini<-now()
set.seed(123)
ctreeFit <- train(
  formula,
  data = whole_data$train,
  method = "ctree2",
  preProc = NULL, 
  tuneGrid = tuneGrid,
  trControl = ctrl,
  metric = "RMSE"
)
print(now()-ini)


ctreeFit 


# inspecting the most relevant features: 
ctreeFit$results
ctreeFit$bestTune
ctreeFit$finalModel

# we can access the K-fold validation predictions
str(ctreeFit$pred)
pred_cv<-data.table(ctreeFit$pred)

# and visualize a CV summary using the established metric
plot(ctreeFit)


# we save the train object
saveRDS(ctreeFit,'ctreeFit_auto.RData')

# 4. Fit the model to all train data 
# (note: as maxdepth -> inf, the RSME shrinks, so just use mincriterion =0.8)
library(party)
ini<-now()
finalmodel<-ctree(formula, data=whole_data$train,
                  control = ctree_control(mincriterion=0.80))

print(now()-ini)

saveRDS(finalmodel,'ctree_finalmodel.RData')

finalmodel <- readRDS('rdata/ctree_finalmodel.RData')

# 5. Predict on train (fit) and test data
pred_train<-predict(finalmodel, whole_data$train)
ctree_mape_train <- mape(real = whole_data$train$Purchase, predicted = pred_train)

pred_test<-predict(finalmodel, whole_data$test)
ctree_mape_test<- mape(real = whole_data$test$Purchase, predicted = pred_test)


#### rf ####

# 1. Define our grid of hyperparameters to tune
tuneGrid=data.table(expand.grid(mtry=c(5,6,7,8,9,10,13,15),
                                splitrule='variance',
                                min.node.size=c(2,5,10)))
# check dimensions and grid
dim(tuneGrid)
tuneGrid


# 2. Define the validation squema
ctrl <- trainControl(
  method = "cv",
  number = 3,
  savePredictions=TRUE
)

# 3. Train the model (with multiple cores)
library(doSNOW)
library(parallel)

numberofcores = detectCores()-1  # review number of cores available (minus 1)
numberofcores

cl <- makeCluster(numberofcores, type = "SOCK", outfile="log.txt")
# Register cluster so that caret will know to train in parallel.
registerDoSNOW(cl)

ini<-now()
set.seed(123)
rangerFit <- train(
  formula,
  data = whole_data$train,
  method = "ranger", num.trees=1000,
  preProc = NULL, 
  tuneGrid = tuneGrid,
  trControl = ctrl,
  metric = "RMSE"
)
print(now()-ini)

stopCluster(cl)

rangerFit


# inspecting the most relevant features: 
rangerFit$results
rangerFit$bestTune
rangerFit$finalModel

# we can access the K-fold validation predictions
str(rangerFit$pred)
pred_cv<-data.table(rangerFit$pred)
pred_cv[Resample=='Fold1'&mtry==5&min.node.size==5]

# and visualize a CV summary using the established metric
plot(rangerFit)

# we save the train object
saveRDS(rangerFit,'rangerFit_auto.RData')


# 4. Fit the model to all train data
library(ranger)
ini<-now()
finalmodel<-ranger(formula, data=whole_data$train,num.trees=1000,
                   mtry=8,
                   min.node.size=10)

print(now()-ini)

saveRDS(finalmodel,'rf_finalmodel.RData')

finalmodel<-readRDS('rf_finalmodel.RData')
# 5. Predict on train (fit) and test data
pred_train<-predict(finalmodel, whole_data$train)$predictions
rf_mape_train<- mape(real = whole_data$train$Purchase,
     predicted = pred_train)

pred_test<-predict(finalmodel, whole_data$test)$predictions
rf_mape_test<-mape(real = whole_data$test$Purchase,
     predicted = pred_test)



#### xgboost ####

# 1. Define our grid of hyperparameters to tune
tuneGrid=expand.grid(eta=c(0.01,0.05,0.1, 0.3, 0.5),
                                gamma=c(0, 3, 5),
                                max_depth=c(2, 6, 10, 15, 20),
                                nrounds=c(25,50,75, 100, 125, 130),
                                min_child_weight=1,
                                colsample_bytree=1,
                                subsample=1)
# check dimensions and grid
dim(tuneGrid)
tuneGrid


# 2. Define the validation squema
ctrl <- trainControl(
  method = "cv",
  number = 3,
  savePredictions=TRUE
)

# 3. Train the model

library(doSNOW)
library(parallel)

numberofcores = detectCores()-1  # review number of cores available (minus 1)
numberofcores

cl <- makeCluster(numberofcores, type = "SOCK", outfile="log.txt")
# Register cluster so that caret will know to train in parallel.
registerDoSNOW(cl)

ini<-now()
set.seed(123)
xgbFit <- train(
  formula,
  data = whole_data$train,
  method = "xgbTree",
  preProc = NULL, 
  tuneGrid = tuneGrid,
  trControl = ctrl,
  metric = "RMSE"
)
print(now()-ini)

stopCluster(cl)

xgbFit


# inspecting the most relevant features: 
xgbFit$results
xgbFit$bestTune
xgbFit$finalModel

# we can access the K-fold validation predictions
str(xgbFit$pred)
pred_cv<-data.table(xgbFit$pred)

# and visualize a CV summary using the established metric
plot(xgbFit)

# we save the train object
saveRDS(xgbFit,'xgbFit_auto.RData')


# 4. Fit the model to all train data
library(xgboost)
ini<-now()
finalmodel<-xgboost(booster='gbtree', 
                    data=as.matrix(whole_data$train[, !'Purchase', with=F]),
                    label=whole_data$train$Purchase,
                    eta=0.1,
                    gamma=0,
                    maxdepth=10,
                    nrounds=125)

print(now()-ini)

saveRDS(finalmodel,'xgb_finalmodel.RData')

# 5. Predict on train (fit) and test data
pred_train<-predict(finalmodel, newdata = as.matrix(whole_data$train[, !'Purchase', with=F]), type='response')
xgb_mape_train <- mape(real = whole_data$train$Purchase,
     predicted = pred_train)

pred_test<-predict(finalmodel, newdata = as.matrix(whole_data$test[, !'Purchase', with=F]), type='response')
xgb_mape_test <- mape(real = whole_data$test$Purchase,
     predicted = pred_test)



result<-data.table(method=c('ctree', 'rf','xgb'),
                   mape_train=c(ctree_mape_train, rf_mape_train, xgb_mape_train),
                   mpae_test=c(ctree_mape_test, rf_mape_test, xgb_mape_test))

saveRDS(result, 'rdata/tuned_results.RData')
                   