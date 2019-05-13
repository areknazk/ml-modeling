# models for ML regression baseline feature engineering 

#### Load Data ####
# sourcing scripts
source('load_libraries.R')
source('f_partition.R')
source('regression_metrics.R')

# load baseline feature engineering data and split to train and test sets
# (NA product categories to category 1/2 values, <5% factors as Others)
whole_data<-f_partition(df=fread('data/BlackFriday_train_ready.csv'),
                        test_proportion = 0.2,
                        seed = 872367823)
str(whole_data)

# convert any integer columns to numeric
whole_data<-lapply(whole_data, function(x){
  return(x[, which(sapply(x, is.integer)):=lapply(.SD, as.numeric), .SDcols=sapply(x,is.integer)])
})
str(whole_data)

# define the formula
formula<-as.formula(Purchase~.)   # Purchase against all other variables


#### 1.1 Base R Partitioning Tree ####
# load libraries
library(rpart)
library(partykit)
library(rpart.plot)

# train model
tree_0<-rpart(formula = formula, data = whole_data$train, method = 'anova', model=TRUE)

# predict on test set
test_tree<-predict(tree_0, newdata = whole_data$test,type = 'vector')

# save predictions
df_pred<-whole_data$test[, .(id=1:.N,Purchase, test_tree)]
str(df_pred)

# plot predictions against real values
ggplot(df_pred, aes(x=df_pred$Purchase, y = df_pred$test_tree))+
  geom_point()

# generate test statistics
rmse_tree<-rmse(real=whole_data$test$Purchase, predicted = test_tree)
mae_tree<-mae(real=whole_data$test$Purchase, predicted = test_tree)
mape_tree<-mape(real=whole_data$test$Purchase, predicted = test_tree)
mape_tree 


#### 1.2 C Partitioning Tree ####
# another type of partitioning trees, based on conditional inference tests

# train model
ctree_0<-ctree(formula, data = whole_data$train)

# predict
test_ctree<-predict(ctree_0, newdata = whole_data$test)

# save predictions
df_pred<-cbind(df_pred, test_ctree)
str(df_pred)

# plot predicitons vs real values
ggplot(df_pred, aes(x=df_pred$Purchase, y = df_pred$test_ctree))+
  geom_point()

# generate test statistics
rmse_ctree<-rmse(real=whole_data$test$Purchase, predicted = test_ctree)
mae_ctree<-mae(real=whole_data$test$Purchase, predicted = test_ctree)
mape_ctree<-mape(real=whole_data$test$Purchase, predicted = test_ctree)
mape_ctree #0.

#### 1.3 Random Forest ####
# load library
library(ranger)

# train model
rf_1<-ranger(formula, whole_data$train)

# predict
test_rf1<-predict(rf_1,whole_data$test)$predictions

# save predictions
df_pred<-cbind(df_pred, test_rf1)
str(df_pred)

# plot predictions vs real values
ggplot(df_pred, aes(x=df_pred$Purchase, y = df_pred$test_rf1))+
  geom_point()

# generate test statistics
rmse_rf<-rmse(real=whole_data$test$Purchase, predicted = test_rf1)
mae_rf<-mae(real=whole_data$test$Purchase, predicted = test_rf1)
mape_rf1<-mape(real=whole_data$test$Purchase, predicted = test_rf1)
mape_rf1 


#### 1.4 Boosting Tree ####
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
mape_xgb 



#### 2.1 Regression with StepWise feature selection ####
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

#### 2.2 Regression with regularization ####
# load library
library(glmnet)

# train model
glmnet_cv<-cv.glmnet(x = data.matrix(whole_data$train[, !'Purchase']),
                     nfolds = 5,
                     y = whole_data$train[['Purchase']],
                     alpha=1,
                     family = 'gaussian',
                     standardize = T)

plot.cv.glmnet(glmnet_cv)

glmnet_cv$lambda.min

# train with min lambda
glmnet_0<-glmnet(x = data.matrix(whole_data$train[, !'Purchase']), 
                 y = whole_data$train[['Purchase']],
                 family = 'gaussian',
                 alpha=1, lambda = glmnet_cv$lambda.min)

glmnet_0

print(glmnet_0)
glmnet_0$beta

# predict
test_glmnet<-predict(glmnet_0, newx = as.matrix(whole_data$test[, !'Purchase']))

# save predictions
df_pred<-cbind(df_pred, test_glmnet=test_glmnet[,1])
str(df_pred)

# plot predictions vs real values
ggplot(df_pred, aes(x=df_pred$Purchase, y = df_pred$test_glmnet))+
  geom_point()

# generate test statistics
rmse_glmnet<-rmse(real=whole_data$test$Purchase, predicted = test_glmnet)
mae_glmnet<-mae(real=whole_data$test$Purchase, predicted = test_glmnet)
mape_glmnet<-mape(real=whole_data$test$Purchase, predicted = test_glmnet)
mape_glmnet 


#### 2.3 Boosting Regression ####
# load library
library(xgboost)

# train model
xgb_reg_0<-xgboost(booster='gblinear',
                   data=as.matrix(whole_data$train[, !'Purchase', with=F]),
                   label=whole_data$train$Purchase,
                   nrounds = 100,
                   objective='reg:linear')
print(xgb_reg_0)

# predict
test_xgb_reg<-predict(xgb_reg_0, newdata = as.matrix(whole_data$test[, !'Purchase', with=F]), type='response')

# save predictions
df_pred<-cbind(df_pred, test_xgb_reg)
str(df_pred)

# plot predicted vs real values
ggplot(df_pred, aes(x=df_pred$Purchase, y = df_pred$test_xgb_reg))+
  geom_point()

# generate test statisitics
rmse_xgb_reg<-rmse(real=whole_data$test$Purchase, predicted = test_xgb_reg)
mae_xgb_reg<-mae(real=whole_data$test$Purchase, predicted = test_xgb_reg)
mape_xgb_reg<-mape(real=whole_data$test$Purchase, predicted = test_xgb_reg)
mape_xgb_reg




#### Model Evaluation ####
result<-data.table(method=c('tree','ctree', 'rf1','xgb','lm','glmnet','xgb_reg'),
                   rmse=sapply(df_pred[,!c('Purchase','id')],function(x) return(rmse(real=df_pred$Purchase, predicted=x))),
                   mae=sapply(df_pred[,!c('Purchase','id')],function(x) return(mae(real=df_pred$Purchase, predicted=x))),
                   mape=sapply(df_pred[,!c('Purchase','id')],function(x) return(mape(real=df_pred$Purchase, predicted=x))))


result


result[which.min(result$mape)]

# plotting results metrics
ggplot(result, aes(x=method, y=mape))+geom_bar(stat='identity')
ggplot(result, aes(x=method, y=rmse))+geom_bar(stat='identity')
ggplot(result, aes(x=method, y=mae))+geom_bar(stat='identity')

