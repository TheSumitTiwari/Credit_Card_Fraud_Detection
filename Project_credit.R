# Credit card Fraud Detection Project.



#install packages

install.packages("ranger")
install.packages("caret")
install.packages("data.table")
library("ranger")
library("caret")
library("data.table")

#importing the data set
creditdata=read.csv("C:/Users/SUMIT TIWARI/Documents/BOOKS/important/Projects/Credit_card_fraud_detection/project/creditcard.csv")
creditdata

#Structure of a data set
str(creditdata)

dim(creditdata)
head(creditdata,10)
tail(creditdata,10)

#Summary
summary(creditdata)
summary(creditdata$Amount)
sd(creditdata$Amount)
names(creditdata)

#get distribution of fraud and legit transaction
table(creditdata$Class)

#get percentage 
prop.table(table(creditdata$Class))
summary(creditdata$Class)

#count the missing value
sum(is.na(creditdata))
#---------------------------------------------------------------------------------------------
#pie chart

labels <-c("legit","fraud")
labels <-paste(labels,round(100*prop.table(table(creditdata$Class)),2))
labels <-paste0(labels,"%")
pie(table(creditdata$Class),labels,col=c("red","yellow"),main="pie chart of transaction")

#---------------------------------------------------------------------------------------------
#co-relation plot

library(corrplot)
creditdata$Class <- as.numeric(creditdata$Class)
corr_plot <- corrplot(cor(creditdata[,-1]), method = "circle", type = "upper")

#---------------------------------------------------------------------------------------------
#transaction amount by legit and fraud person

classes <-factor(creditdata$Class,levels = c(0,1)) #classes as a factor to use in plot

common_theme <- theme(plot.title = element_text(hjust = 0.5, face = "bold"))
p <- ggplot(creditdata, aes(x = classes, y = Amount)) + geom_boxplot() + ggtitle("Distribution of transaction amount by class") + common_theme
print(p)

#---------------------------------------------------------------------------------------------
#normalize Amount

normalize <- function(x){
  return((x - mean(x, na.rm = TRUE))/sd(x, na.rm = TRUE))
}
creditdata$Amount <- normalize(creditdata$Amount)

#----------------------------------------------------------------------------------------------
#t-SNE model to check any pattern in data set

install.packages("Rtsne")
library(Rtsne)
tsne_subset <- 1:as.integer(0.1*nrow(creditdata))
tsne <- Rtsne(creditdata[tsne_subset,-1,-31], perplexity = 20, theta = 0.5, pca = F, verbose = T, max_iter = 500, check_duplicates = F)

classes <- as.factor(creditdata$Class[tsne_subset])
tsne_mat <- as.data.frame(tsne$Y)
ggplot(tsne_mat, aes(x = V1, y = V2)) + geom_point(aes(color = classes)) + theme_minimal() + 
        common_theme + ggtitle("t-SNE visualisation of transactions") + scale_color_manual(values = c("#E69F00", "#56B4E9"))
#---------------------------------------------------------------------------------------------
#Data Manipulation

newData=creditdata[,-c(1)]
head(newData)

#---------------------------------------------------------------------------------------------
#Data Modeling

install.packages("caTools")
library("caTools")
set.seed(123)   

creditdata$Class <- as.numeric(creditdata$Class)

datasample=sample.split(newData$Class,SplitRatio = 0.80)
traindata=subset(newData,datasample==T)
testdata=subset(newData,datasample==F)

#----------------------------------------------------------------------------------------------
#Fitting Logistic Regression Model

dim(traindata)
dim(testdata)
summary(testdata)

Logistic_model=glm(Class~.,traindata,family = binomial())
summary(Logistic_model)
plot(Logistic_model)

install.packages("pROC")
library("pROC")
lr.predict=predict(Logistic_model,testdata,probability=TRUE)

lr.predict.rd <- ifelse(lr.predict > 0.5, 1, 0)

install.packages("e1071")
library("e1071")

conf_mat <- confusionMatrix(factor(testdata[,30],levels = c(0,1)), factor(lr.predict.rd,levels = c(0,1)))
print(conf_mat)
fourfoldplot(conf_mat$table, color = c("#CC6666", "#99CC99"), conf.level = 0, margin = 1, main = "Confusion Matrix")

auc.gbm=roc(testdata$Class,lr.predict,plot = TRUE,col="blue")
plot(auc.gbm, main = paste0("AUC: ", round(pROC::auc(auc.gbm), 3)))

#--------------------------------------------------------------------------------------------
#Decision Tree Model

install.packages("rpart")
install.packages("rpart.plot")
library("rpart")
library("rpart.plot")

DecisionTreeModel=rpart(Class~.,traindata,method = 'class')
rpart.plot(DecisionTreeModel)

predictedval=predict(DecisionTreeModel,testdata,type='class')
probability=predict(DecisionTreeModel,testdata,type = 'prob')

conf_mat <- confusionMatrix(factor(testdata[,30],levels = c(0,1)), predictedval)
print(conf_mat)
fourfoldplot(conf_mat$table, color = c("#CC6666", "#99CC99"), conf.level = 0, margin = 1, main = "Confusion Matrix")

auc.gbm=roc(testdata$Class,as.numeric(predictedval),plot = TRUE,col="blue")
plot(auc.gbm, main = paste0("AUC: ", round(pROC::auc(auc.gbm), 3)))

#---------------------------------------------------------------------------------------------
#Artificial Neural Network

install.packages("neuralnet")
library("neuralnet")
AModel=neuralnet(Class~.,traindata,linear.output = FALSE)
plot(AModel)

predANN=compute(AModel,testdata)
resultANN=predANN$net.result

resultANN=ifelse(resultANN>0.5,1,0)
conf_mat <- confusionMatrix(factor(testdata[,30],levels = c(0,1)), factor(resultANN,levels = c(0,1)))
print(conf_mat)
fourfoldplot(conf_mat$table , color = c("#CC6666", "#99CC99"), conf.level = 0, margin = 1, main = "Confusion Matrix")

auc.gbm=roc(testdata$Class,as.numeric(resultANN),plot = TRUE,col="blue")
plot(auc.gbm, main = paste0("AUC: ", round(pROC::auc(auc.gbm), 3)))
#---------------------------------------------------------------------------------------------
