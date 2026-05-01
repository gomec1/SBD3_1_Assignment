# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# --------------------------- CLASSIFICATION IN R -------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# In this script, we learn how we can use R to train and test different models for a classification task.
# Specifically, we will learn how to apply the following algorithms:
#  - Logistic Regression 
#  - Simple Decision Tree
#  - Random Forest
#  - NEW: XGBoost
#  - NEW: Deep Neural Network (MLP) 

# We start by calling all the libraries that we are going to use. 
library(readr)
library(dplyr)
library(caret)
library(DescTools)
library(rsample)
library(pROC)
library(ROCR)
library(randomForest)
library(rpart) 
library(rpart.plot) 
library (dlookr)
options(scipen=999)


#clear environment
rm(list = ls())


#################################################
##### Step 1: Data – Import and Data Check ######
#################################################
## Import data
# We set the working directory and we call the data that we are going to use. Please import the csv file "Employee-Attrition.csv".
setwd("~/Library/CloudStorage/OneDrive-SharedLibraries-BernerFachhochschule/AIRS - General/Teaching/SBD3/2026/Week 5 - Supervised Learning II/New practical.")
Employee_Attrition_cleaned <- read_csv("Employee-Attrition_cleaned.csv")
data <- Employee_Attrition_cleaned # We make a copy from the original dataset and work on the copy

# Checking dimensions and structure of the data
dim(data)
str(data)
head(data)

# We check the summary of all variables included. Here might be able to identify some data quality issues.
summary(data)

# Examine if there are missing values
diagnose(data)

# Y: Check how represented each class is
PercTable(data$Attrition)

###############################################################
######## Step 2: Split data into train and test set  ##########
###############################################################

## Split the data into training and testing sets. Here, we use 70% of the observations for training and 30% for testing.
# To ensure that the target variable is represented equally in both data sets, we apply a stratified sampling approach.
set.seed(7) # Set random seed to make results reproducible
data$Attrition = as.factor(data$Attrition)
split  <- initial_split(data, prop = 0.7)
data.train  <- training(split)
data.test   <- testing(split)

PercTable(data.train$Attrition) # check the distribution of the target variable in train data
PercTable(data.test$Attrition) # check the distribution of the target variable in test data

####################################################################
## Step 3: Train and evaluate your model(s) [BASIC IMPLEMENTAION]##
##################################################################

##################################
### Logistic Regression ##########
##################################

# Train a model explaining Attrition (i.e. Y), specify independent variables (i.e. Xs)
logistic_model <- glm(Attrition ~ Age + 
                                DistanceFromHome + 
                                MonthlyIncome + 
                                JobSatisfaction, 
                                data=data.train,
                                family=binomial())

# Print results of the model
summary(logistic_model)

# Make predictions on the test data
data.test$pred_score_lg <- predict(logistic_model, type='response', data.test)

# Predict churn if probability is greater than 50%
data.test$pred_lg <- ifelse(data.test$pred_score_lg > 0.5, "Yes", "No")
data.test$pred_lg <- as.factor (data.test$pred_lg)

# Examine the confusion matrix
table(data.test$pred_lg, data.test$Attrition)

# Compute the accuracy on the test dataset
mean(data.test$pred_lg == data.test$Attrition)


##################################
### Simple Classification Tree ###
##################################

## Train a model explaining Attrition (i.e. Y), specify independent variables (i.e. Xs)
# Specify: maximum tree depth and the complexity parameter (stoping parameter)
tree_model <- rpart(Attrition ~ Age + 
                                DistanceFromHome + 
                                MonthlyIncome + 
                                JobSatisfaction, 
                                data=data.train, 
                                method = "class", control = rpart.control(cp = 0.001, maxdepth =4))

# Plot the decision tree
rpart.plot(tree_model, type=5)

# Make predictions on the test data
data.test$pred_DT <- predict(tree_model, type='class', data.test)

# Examine the confusion matrix
table(data.test$pred_DT, data.test$Attrition)

# Compute the accuracy on the test dataset
mean(data.test$pred_DT == data.test$Attrition)


##################################
### Random Forest ################
##################################

### Train a model explaining Attrition (i.e. Y), specify independent variables (i.e. Xs)
## ntree defines the number of trees to be generated
# mtry is the number of features used in the construction of each tree. Default value = sqrt(number of features)
RF_model <- randomForest(Attrition ~ Age + 
                                        DistanceFromHome + 
                                        MonthlyIncome + 
                                        JobSatisfaction, 
                                        data=data.train, ntree=10 , mtry= 4, importance=TRUE)

# Make predictions on the test data
data.test$pred_RF <- predict(RF_model, type='class', data.test)

# Examine the confusion matrix
table(data.test$pred_RF, data.test$Attrition)

# Compute the accuracy on the test dataset
mean(data.test$pred_RF == data.test$Attrition)

# Plot variable importance
varImpPlot(RF_model)


# EXTENSION! 

##################################
### XGBoost ######################
##################################

# Load package
#install.packages("xgboost")
library(xgboost)

# XGBoost requires the target variable to be numeric: No = 0, Yes = 1
data.train$Attrition_xgb <- ifelse(data.train$Attrition == "Yes", 1, 0)
data.test$Attrition_xgb  <- ifelse(data.test$Attrition == "Yes", 1, 0)

# Select predictors and create model matrices
x_train <- as.matrix(data.train[, c("Age",
                                    "DistanceFromHome",
                                    "MonthlyIncome",
                                    "JobSatisfaction")])

x_test <- as.matrix(data.test[, c("Age",
                                  "DistanceFromHome",
                                  "MonthlyIncome",
                                  "JobSatisfaction")])
# Add the target variable 
y_train <- data.train$Attrition_xgb
y_test  <- data.test$Attrition_xgb

# Convert data into DMatrix format (this is the optimized structure for xgboost)
dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest  <- xgb.DMatrix(data = x_test, label = y_test)

#################################################
### Train model #################################
#################################################

set.seed(7)

params <- list(
  objective = "binary:logistic", # This defines what type of prediction problem the model is solving. 
  eval_metric = "auc",           # This defines the evaluation metric used to assess the performance of the model during training.
  eta = 0.05,                    # learning rate means how much the model weights are updated during training. A smaller value makes the model more robust to overfitting but requires more trees to be built.
  max_depth = 20,                # tree depth controls how complex the individual trees are. A deeper tree can capture more complex relationships but is more likely to overfit.
  min_child_weight = 1,          # This parameter controls the minimum sum of instance weight (hessian) needed in a child. It is used to control overfitting. Higher values prevent the model from learning relations which might be highly specific to the particular sample selected for a tree.
  subsample = 0.8,               # row sampling means that each tree is built using a random sample of the training data. This can help to prevent overfitting and improve the generalization of the model.
  colsample_bytree = 0.8,        # feature sampling means that each tree is built using a random sample of the features. This can also help to prevent overfitting and improve the generalization of the model.
  gamma = 0.1                    # This parameter specifies the minimum loss reduction required to make a further partition on a leaf node of the tree. It is used to control overfitting. Higher values lead to more conservative models.
)

xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 1000,              # the maximum number of boosting iterations
  watchlist = list(train=dtrain,test=dtest),
  early_stopping_rounds = 20,  # Stop training if the evaluation metric on the validation set does not improve for 20 rounds.
  verbose = 1                  # print evaluation metrics
)

#################################################
### Predictions #################################
#################################################

data.test$pred_score_xgb <- predict(xgb_model,newdata=dtest)

data.test$pred_XGB <- ifelse(data.test$pred_score_xgb > 0.5,"Yes","No")
data.test$pred_XGB <- as.factor(data.test$pred_XGB)

#################################################
### Evaluation ##################################
#################################################

table(data.test$pred_XGB,data.test$Attrition)

mean(data.test$pred_XGB == data.test$Attrition)

#################################################
### Variable importance #########################
#################################################

# XGBoost variable importance is calculated by measuring how much each feature 
# contributes to improving the model’s objective function across all tree splits 
# (Gain), how many observations are affected by those splits (Cover), and how 
# often the feature is used in splits (Frequency).

importance_matrix <- xgb.importance(model=xgb_model,
                                    feature_names=colnames(x_train))

print(importance_matrix)

xgb.plot.importance(importance_matrix)


##################################
### Deep Neural Network (MLP) ####
##################################

# Before proceeding - open Script 00_setup_python_once.R and run it (!)

set.seed(7)
tensorflow::set_random_seed(7)

predictor_names <- c(
  "Age",
  "DistanceFromHome",
  "MonthlyIncome",
  "JobSatisfaction"
)

# Ensure numeric predictors
x_train <- data.train[, predictor_names]
x_test  <- data.test[, predictor_names]

# Scale using training statistics only
# Scaling is important for a Multi-Layer Perceptron (MLP) because neural networks
# are trained using gradient-based optimization, and large differences in feature 
# scales can make training unstable or very slow.
train_means <- sapply(x_train, mean, na.rm = TRUE)
train_sds   <- sapply(x_train, sd, na.rm = TRUE)

x_train <- scale(x_train, center = train_means, scale = train_sds)
x_test  <- scale(x_test, center = train_means, scale = train_sds)

x_train <- as.matrix(x_train)
x_test  <- as.matrix(x_test)

y_train <- ifelse(data.train$Attrition == "Yes", 1, 0)
y_test  <- ifelse(data.test$Attrition == "Yes", 1, 0)

# Define the MLP architecture: 
# - Input layer: number of features (4 in our case)
# - Hidden layers: 3 hidden layers with 32, 16, and 8
#  neurons respectively, and ReLU activation function.
# - Output layer: 1 neuron with sigmoid activation function 
# (for binary classification).
inputs <- keras_input(shape = c(ncol(x_train)))

outputs <- inputs %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 8, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

mlp_model <- keras_model(inputs = inputs, outputs = outputs)


# # Adam is the method the neural network uses to update its weights efficiently 
# after each batch so that prediction errors gradually decrease during training.
mlp_model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),  
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

# Early stopping is a regularization technique that helps prevent overfitting by 
# monitoring the model's performance on a validation set during training. 
# If the model's performance does not improve for a specified number of epochs (patience), 
# training is stopped, and the best weights are restored.
early_stop <- callback_early_stopping(
  monitor = "val_loss",
  patience = 20,
  restore_best_weights = TRUE
)

# We do not process all observations at one go but rather in batches. Batch size 
# is the number of samples processed before the model's internal
# parameters are updated. A smaller batch size can lead to more stable training 
# but may take longer to converge, while a larger batch size can speed up training 
# but may lead to less stable updates. Here we use a batch size of 32. This means
# that the model's weights will be updated after processing every 32 samples from the training data.
# Epochs is the number of complete passes through the training dataset. 
# Here we set it to 100, but with early stopping, the model will stop training 
# if the validation loss does not improve for 20 consecutive epochs.
history <- mlp_model %>% fit(
  x = x_train,
  y = y_train,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2,
  callbacks = list(early_stop),
  verbose = 1
)

pred_prob_mlp <- predict(mlp_model, x_test)
pred_class_mlp <- ifelse(pred_prob_mlp > 0.5, "Yes", "No")
pred_class_mlp <- factor(pred_class_mlp, levels = c("No", "Yes"))

actual_label <- factor(data.test$Attrition, levels = c("No", "Yes"))

table(pred_class_mlp, actual_label)
mean(pred_class_mlp == actual_label)
confusionMatrix(pred_class_mlp, actual_label, positive = "Yes")
