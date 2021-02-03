#libraries
library(DMwR)
library(readr)
library(dplyr)
library(caret)
library(truncnorm)
library(caretEnsemble)
library(doParallel)
library(xgboost)

#import data and combine:
train <- read_csv("C:/Users/Saniya and Family/Downloads/train (1).csv")
test <- read_csv("C:/Users/Saniya and Family/Downloads/test (1).csv")
final <- read_csv("C:/Users/Saniya and Family/Downloads/gender_submission.csv")

test$Survived <- final$Survived
data <- rbind(train,test)
data <- data[,-1] #remove id
data <- data[,-3]#remove name

#NA values:
names(which(sapply(data, anyNA)))

#Embarked
median(data$Embarked, na.rm = T)
data$Embarked[is.na(data$Embarked)] <- "S"

#Fare:
data$Fare[is.na(data$Fare)] <- median(data$Fare, na.rm = T)

#Age NA
range(data$Age, na.rm = T)
age_sum <- data %>% group_by(Pclass, Sex) %>% summarize(mean=mean(Age,na.rm = T),
                                                        min = min(Age,na.rm = T),
                                                        max = max(Age, na.rm = T))

no_NA <- data %>% group_by(Pclass,Sex) %>% summarize(count=sum(is.na(Age)))

k <- 1
for ( i in 1:3) {
  for (j in c("female","male")) {
    data$Age[data$Pclass==i & data$Sex==j & is.na(data$Age)] <- rtruncnorm(no_NA[[k,3]],
                                                                           a=age_sum[[k,5]],
                                                                           mean=age_sum[[k,3]],
                                                                           sd=age_sum[[k,4]])
    k=k+1
  }
}

#Categorical values:
data$Sex[data$Sex=="male"] <- 1
data$Sex[data$Sex=="female"] <- 2
data$Sex <- as.numeric(data$Sex)

data$Embarked[data$Embarked=="C"] <- 1
data$Embarked[data$Embarked=="Q"] <- 2
data$Embarked[data$Embarked=="S"] <- 3
data$Embarked <- as.numeric(data$Embarked)

data <- data[,-c(7,9)] #remove cabin and ticket

str(data)

data$Pclass <- as.factor(data$Pclass)
data$Sex <- as.factor(data$Sex)
data$SibSp <- as.factor(data$SibSp)
data$Parch <- as.factor(data$Parch)
data$Embarked <- as.factor(data$Embarked)

datanum <- select_if(data,is.numeric)
names(datanum)

#train and test
trainD <- data[c(1:891),]
testD <- data[c(892:1309),]



preProcValues    <- preProcess(as.data.frame(trainD[,names(trainD) %in% names(datanum)]), method = c("nzv","BoxCox","center", "scale"))
trainTransformed <- predict(preProcValues, as.data.frame(trainD))
testTransformed  <- predict(preProcValues, as.data.frame(testD))

#removing y
testTransformed <- testTransformed[,-1]
trainTransformed <- trainTransformed[,-1]
trainTransformed$Survived <- as.character(train$Survived)
trainTransformed$Survived[trainTransformed$Survived==0] <- "X0"
trainTransformed$Survived[trainTransformed$Survived==1] <- "X1"
trainTransformed$Survived <- as.factor(trainTransformed$Survived)
levels(trainTransformed$Survived)


#feature Plot
featurePlot(x=trainTransformed[,-8], y=trainTransformed$Survived, between= list(x=1, y=1),type=c("g","p","smooth"))

#formula:
modformula <- as.formula(paste("Class ~. "))

#models
registerDoParallel(4)
getDoParWorkers()

train_sample <- upSample(x=trainTransformed[,-8], y=trainTransformed[,8])

ctrl <- trainControl(method="cv", number=5,  savePredictions="final", summaryFunction = twoClassSummary, 
                     classProbs = T,allowParallel=T ,index=createFolds(trainTransformed$Survived,5))

model_list <- caretList( modformula, data=train_sample, trControl=ctrl,
                         tuneList = list(
                           modxgb=caretModelSpec(method="xgbTree", 
                                                 tuneGrid=expand.grid(eta=c(0.4), 
                                                                      gamma=0,
                                                                      max_depth=c(10),
                                                                      nrounds=c(1000),
                                                                      colsample_bytree=c(0.5),
                                                                      min_child_weight=c(4),
                                                                      subsample=c(0.5))),
                           modnb =caretModelSpec(method="naive_bayes",
                                                  tuneGrid=  expand.grid(usekernel = c(TRUE, FALSE),
                                                                         laplace = c(0, 0.5, 1), 
                                                                         adjust = c(0.75, 1, 1.25, 1.5))),
                           modgbm =caretModelSpec(method="gbm",
                                                  tuneGrid=expand.grid(n.trees = seq(100,1000,by=400),
                                                                       interaction.depth = c(1,2),
                                                                       shrinkage = c(0.01,0.1),
                                                                       n.minobsinnode = c(10,30,50))),
                           modrf =caretModelSpec(method="parRF",
                                                 tuneGrid=expand.grid(mtry=c(2,8,12,16,20,30))),
                           modrdg=caretModelSpec(method="glmnet",
                                                 tuneGrid=expand.grid(alpha = 0, 
                                                                      lambda = seq(0.01, 0.9, by=0.01)))
                          
                         ))


#predictions
ensemble <- caretEnsemble(model_list, trControl=ctrl, metric="ROC")

pe <- predict(ensemble, trainTransformed)
pxg <- predict(model_list$modxgb, trainTransformed)
pnb<- predict(model_list$modnb, trainTransformed)
pgbm <- predict(model_list$modgbm, trainTransformed)
prf<- predict(model_list$modrf, trainTransformed)
prdg <- predict(model_list$modrdg, trainTransformed) 

confusionMatrix(trainTransformed$Survived, pe)
#compare all the values and use the one with best accuracy

pe <- predict(ensemble,testTransformed)
pxg <- predict(model_list$modxgb, testTransformed)
pnb<- predict(model_list$modnb, testTransformed)
pgbm <- predict(model_list$modgbm, testTransformed)
prf<- predict(model_list$modrf, testTransformed)
prdg <- predict(model_list$modrdg, testTransformed) 


final$Survived <- as.character(pe)
final$Survived[final$Survived=="X0"] <- 0
final$Survived[final$Survived=="X1"] <- 1
final$Survived <- as.factor(final$Survived)

write.csv(final,"titanic.csv", row.names = F)











