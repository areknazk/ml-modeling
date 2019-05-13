# models for ML regression version 3 feature engineering 

#### Load Data ####
# sourcing scripts
source('load_libraries.R')
source('f_partition.R')
source('regression_metrics.R')

# load version 3 feature engineering data and split to train and test sets
# (NA product categories to 0, <5% factors as either high-ourchase-others or low-purchase-others)
whole_data<-f_partition(df=fread('data/BlackFriday_train_ready3.csv'),
                        test_proportion = 0.2,
                        seed = 872367823)
str(whole_data)

# convert any integer columns to numeric
whole_data<-lapply(whole_data, function(x){
  return(x[, which(sapply(x, is.integer)):=lapply(.SD, as.numeric), .SDcols=sapply(x,is.integer)])
})
str(whole_data)

# define the formula
formula<-as.formula(Purchase~ .)  

#### C Partitioning Tree ####
# load library
library(party)

# train model
ctree_0<-ctree(formula, data = whole_data$train)

# predict
test_ctree<-predict(ctree_0, newdata = whole_data$test)

# save predictions
df_pred<-whole_data$test[, .(id=1:.N,Purchase, test_ctree)]
str(df_pred)

# plot predicted vs real values
ggplot(df_pred, aes(x=df_pred$Purchase, y = df_pred$test_ctree))+
  geom_point()

# generate test statistics
rmse_ctree<-rmse(real=whole_data$test$Purchase, predicted = test_ctree)
mae_ctree<-mae(real=whole_data$test$Purchase, predicted = test_ctree)
mape_ctree<-mape(real=whole_data$test$Purchase, predicted = test_ctree)
mape_ctree 


#### Random Forest ####
# load library
library(ranger)

# rf has trouble with the "+" in "Stay_In_Current_City_Years_4+"
whole_data$train$Stay_In_Current_City_Years_4<-whole_data$train$`Stay_In_Current_City_Years_4+`
whole_data$train[, c("Stay_In_Current_City_Years_4+"):=NULL]
whole_data$test$Stay_In_Current_City_Years_4<-whole_data$test$`Stay_In_Current_City_Years_4+`
whole_data$test[, c("Stay_In_Current_City_Years_4+"):=NULL]

# train model
rf_1<-ranger(formula, whole_data$train)

# predict
test_rf1<-predict(rf_1,whole_data$test)$predictions

# save predictions
df_pred<-cbind(df_pred, test_rf1)
str(df_pred)

# plot predicted vs real values
ggplot(df_pred, aes(x=df_pred$Purchase, y = df_pred$test_rf1))+
  geom_point()

# generate test statistics
rmse_rf<-rmse(real=whole_data$test$Purchase, predicted = test_rf1)
mae_rf<-mae(real=whole_data$test$Purchase, predicted = test_rf1)
mape_rf1<-mape(real=whole_data$test$Purchase, predicted = test_rf1)
mape_rf1 


#### Boosting Tree ####
# load library
library(xgboost)

# for this algorithm we need to convert data to a matrix first, and then train
xgb_0<-xgboost(booster='gbtree',
               data=as.matrix(whole_data$train[, !'Purchase', with=F]),
               label=whole_data$train$Purchase,
               nrounds = 50,
               objective='reg:linear')

# predict
test_xgb<-predict(xgb_0, newdata = as.matrix(whole_data$test[, !'Purchase', with=F]), type='response')

# save predictions
df_pred<-cbind(df_pred, test_xgb)
str(df_pred)

# plot predicted vs real values
ggplot(df_pred, aes(x=df_pred$Purchase, y = df_pred$test_xgb))+
  geom_point()

# generate test statistics
rmse_xgb<-rmse(real=whole_data$test$Purchase, predicted = test_xgb)
mae_xgb<-mae(real=whole_data$test$Purchase, predicted = test_xgb)
mape_xgb<-mape(real=whole_data$test$Purchase, predicted = test_xgb)
mape_xgb #0.6267253



#### Regression with StepWise feature selection ####
# load library
library(MASS)

# train model
lm_0<-stepAIC(lm(formula = formula, 
                 data=whole_data$train),
              trace=F)
summary(lm_0)

# predict
test_lm<-predict(lm_0, newdata = whole_data$test)

# save predictions
df_pred<-cbind(df_pred, test_lm)
str(df_pred)

# plot predicted vs real values
ggplot(df_pred, aes(x=df_pred$Purchase, y = df_pred$test_lm))+
  geom_point()

# generate test statistics
rmse_lm<-rmse(real=whole_data$test$Purchase, predicted = test_lm)
mae_lm<-mae(real=whole_data$test$Purchase, predicted = test_lm)
mape_lm<-mape(real=whole_data$test$Purchase, predicted = test_lm)
mape_lm 


#### Model Evaluation ####

result<-data.table(method=c('ctree', 'rf1','xgb','lm'),
                   rmse=sapply(df_pred[,!c('Purchase','id')],function(x) return(rmse(real=df_pred$Purchase, predicted=x))),
                   mae=sapply(df_pred[,!c('Purchase','id')],function(x) return(mae(real=df_pred$Purchase, predicted=x))),
                   mape=sapply(df_pred[,!c('Purchase','id')],function(x) return(mape(real=df_pred$Purchase, predicted=x))))
result
result[which.min(result$mape)]

# plotting results metrics
ggplot(result, aes(x=method, y=mape))+geom_bar(stat='identity')
ggplot(result, aes(x=method, y=rmse))+geom_bar(stat='identity')
ggplot(result, aes(x=method, y=mae))+geom_bar(stat='identity')