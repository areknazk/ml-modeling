## In this script we use the final random forest model to make final predictions on the test data 
source('load_libraries.R')

# read in data and model
final_model <- readRDS('rdata/rf_finalmodel.RData')
test_data <- fread('data/BlackFriday_test_ready1.csv')

# fix the column name to be readable
test_data$Stay_In_Current_City_Years_4<-test_data$`Stay_In_Current_City_Years_4+`
test_data[, c("Stay_In_Current_City_Years_4+"):=NULL]

library(ranger)

# run predicctions
pred<-predict(final_model, test_data)$predictions

# add prediction column to the csv
test_data$Purchase <- pred

# save the new csv
fwrite(test_data, file = "data/black_friday_test_w_prediction.csv")
