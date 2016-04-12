#The following R packages are required to run the model
require(xgboost)
require(methods)

#Convert the Target Lablels to Integers. XGBoost takes only numerical classes from 0 to n.
train = read.csv('D:/Google Drive/CIS508 Data Mining-1/Team Project/train.csv',header=TRUE,stringsAsFactors = F)
test = read.csv('D:/Google Drive/CIS508 Data Mining-1/Team Project/test.csv',header=TRUE,stringsAsFactors = F)
#Remove the ID Column as it can used while training the data
train = train[,-1]
test = test[,-1]

#Convert the Target Lablels to Integers. XGBoost takes only numerical classes from 0 to n.
y = train[,ncol(train)]
y = gsub('Class_','',y)
y = as.integer(y)-1 


x = rbind(train[,-ncol(train)],test)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)

# XGBOOST Parameters set for multiclass classfication problem with logloss as measure of error as per the problem in Kaggle
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "nthread" = 8)

# Cross-Validation Cycles
cv.nround = 100
bst.cv = xgb.cv(param=param, data = x[trind,], label = y, 
                nfold = 6, nrounds=cv.nround)

# XGBOOST Model Training
nround = 100
bst = xgboost(param=param, data = x[trind,], label = y, nrounds=nround, max.depth=3)

# Prediction of Target Labels on Test Data
pred = predict(bst,x[teind,])
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)

# Retrieve Output as per the desired Kaggle Format
pred = format(pred, digits=2,scientific=F)
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
write.csv(pred,file='D:/Google Drive/CIS508 Data Mining-1/Team Project/submission.csv', quote=FALSE,row.names=FALSE)