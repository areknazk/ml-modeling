# models for ML regression with interactions between features 

#### Load Data ####
# sourcing scripts
source('load_libraries.R')
source('f_partition.R')
source('regression_metrics.R')

# load version 1 feature engineering data and split to train and test sets
# (NA product categories to category 1/2 values, <5% factors as either high-ourchase-others or low-purchase-others)
whole_data<-f_partition(df=fread('data/BlackFriday_train_ready1.csv'),
                        test_proportion = 0.2,
                        seed = 872367823)
str(whole_data)

# convert any integer columns to numeric
whole_data<-lapply(whole_data, function(x){
  return(x[, which(sapply(x, is.integer)):=lapply(.SD, as.numeric), .SDcols=sapply(x,is.integer)])
})
str(whole_data)

# define the formula (with interactions)
formula<-as.formula(Purchase~.*.)   # Purchase against all other variables with interaction terms

# Note: tree based models already include interaction terms by construction, so we will only test interactions on the linear regression model

#### Regression with StepWise feature selection ####
# load library
library(MASS)

# sampling may be required for memory capabilities
sample_index <- sample(nrow(whole_data$train), floor(nrow(whole_data$train)*0.5), replace = FALSE)

# train model
lm_0<-stepAIC(lm(formula = formula, 
                 data=whole_data$train[sample_index]),
              trace=F)

# save model
saveRDS(lm_0,'lm_0_with_interactions.RData')

summary(lm_0)

# predict
test_lm<-predict(lm_0, newdata = whole_data$test)

# plot predicted vs real values
ggplot(df_pred, aes(x=df_pred$Purchase, y = df_pred$test_lm))+
  geom_point()

# generate test statistics
rmse_lm<-rmse(real=whole_data$test$Purchase, predicted = test_lm)
mae_lm<-mae(real=whole_data$test$Purchase, predicted = test_lm)
mape_lm<-mape(real=whole_data$test$Purchase, predicted = test_lm)
mape_lm 