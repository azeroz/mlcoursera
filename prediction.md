---
title: "Practical Machine Learning Project"
output:
  pdf_document:
    keep_tex: yes
  html_document:
    keep_md: yes
---

### Executive Summary
Our goal is to use data from accelerometers on the belt, forearm, and dumbbell of 6 participants to quantify how well they are doing a particular activity. We will accomplish this by training a prediction model - a random forest classifier - on the accelerometer data.

The data for this project comes from this source: http://groupware.les.inf.puc-rio.br/har.

### Load Options

```r
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(doMC))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(knitr))
suppressPackageStartupMessages(library(scales))

set.seed(8832)
registerDoMC(cores = 4)
```

### Exploratory Analysis and Feature Selection

```r
training_URL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_URL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training <- read.csv(training_URL,na.strings = c("NA",""))
test <- read.csv(test_URL,na.strings = c("NA",""))
```

Remove unrelevant variables that are unlikely to be related to dependent variables.

```r
remove = c('X', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 'cvtd_timestamp', 'new_window', 'num_window')
training.dere <- training[, -which(names(training) %in% remove)]
dim(training.dere)
```

```
## [1] 19622   153
```

Remove NA variables that would not be useful for training the model.

```r
training.omitna <- training.dere[, colSums(is.na(training.dere)) == 0]
dim(training.omitna)
```

```
## [1] 19622    53
```

Split data to training and testing for cross validation, use 70% for the training set, 30% for the testing set.

```r
inTrain <- createDataPartition(y = training.omitna$classe, p = 0.7, list = F)
training <- training.omitna[inTrain, ]
testing <- training.omitna[-inTrain, ]
```

Our training data set has 13737 samples and 53 variables for testing and our testing data set has 5885 samples and 53 variables for testing.

### Fitting Random Forests with Cross Validation
Use 5-fold cross validation and fit the model using the Random Forests algorithm. This should give us a relatively low out of sample error rate.


```r
traincontrol <- trainControl(method = "cv", number = 5)

rf_model <- train(classe ~ ., data = training, method = "rf",
                  trControl = traincontrol,
                  prox = TRUE, allowParallel = TRUE)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## 
## The following object is masked from 'package:dplyr':
## 
##     combine
```

```
## Warning in nominalTrainWorkflow(x = x, y = y, wts = weights, info =
## trainInfo, : There were missing values in resampled performance measures.
```

```r
print(rf_model)
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## 
## Summary of sample sizes: 10990, 10989, 10989, 10991, 10989 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9901716  0.9875648  0.001489277  0.001883924
##   27    0.9911746  0.9888347  0.001202594  0.001520505
##   52    0.9851978  0.9812732  0.003281455  0.004158588
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
print(rf_model$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, proximity = TRUE,      allowParallel = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.68%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3902    1    1    0    2 0.001024066
## B   21 2633    4    0    0 0.009405568
## C    0   16 2369   11    0 0.011268781
## D    0    0   28 2221    3 0.013765542
## E    0    0    3    4 2518 0.002772277
```

With the model having been fit with training data, we use it for predictions on test data.

### Out of Sample Accuracy and Error Estimation
With the model having been fit with training data, we use it for predictions on test data set aside during variable selection. We generate the confusion matrix and estimate the out of sample error rate. The testing data set should be an unbiased estimate of the random forest's prediction accuracy.


```r
# Predict the values for 'Classe' by applying the t rained model to the testing data set.
confMatrix <- confusionMatrix(predict(rf_model, newdata = testing), testing$classe)
confMatrix
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1670    8    0    0    0
##          B    4 1131    6    0    1
##          C    0    0 1019   12    1
##          D    0    0    1  951    1
##          E    0    0    0    1 1079
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9941          
##                  95% CI : (0.9917, 0.9959)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9925          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9976   0.9930   0.9932   0.9865   0.9972
## Specificity            0.9981   0.9977   0.9973   0.9996   0.9998
## Pos Pred Value         0.9952   0.9904   0.9874   0.9979   0.9991
## Neg Pred Value         0.9990   0.9983   0.9986   0.9974   0.9994
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2838   0.1922   0.1732   0.1616   0.1833
## Detection Prevalence   0.2851   0.1941   0.1754   0.1619   0.1835
## Balanced Accuracy      0.9979   0.9953   0.9953   0.9931   0.9985
```

The expected out of sample error rate is 0.983% as the accuracy of the model observed above is 99%. Calculating the out of sample error (the cross-validation estimate is an out-of-sample estimate) we get the value of 0.595%.

### Predict the 20 test cases
Finally, to predict the classe of the testing dataset, we're applying the prediction using the model we've trained and output the results in the respective files as adviced by the instructor:


```r
test_prediction <- predict(rf_model, test)
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(test_prediction)
```
