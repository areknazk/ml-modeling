mae<-function(real, predicted){
  return(mean(abs(real-predicted)))
}

mape<-function(real,predicted){
  return(mean(abs((real-predicted)/real)))
}

rmse<-function(real,predicted){
  return(sqrt(mean((real-predicted)^2)))
}
