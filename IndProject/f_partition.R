f_partition<-function(df,test_proportion=0.2, seed=NULL){
  
  if(!is.null(seed)) set.seed(seed)
  
  train_index<-sample(nrow(df), floor(nrow(df)*(1-test_proportion)), replace = FALSE)
  df_train<-df[train_index]
  df_test<-df[-train_index]
  
  return(list(train=df_train, test=df_test))
}


