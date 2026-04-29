# ============================================================================ #
#   SBD3 | Group Work 1 – "Find Out Your Future Salary"                       #
#   Model: XGBoost (Gradient Boosting Regression)                             #
#   Author: Carlos                                                             #
# ============================================================================ #


# ============================================================================ #
# Step 0: Load required packages
# ============================================================================ #
library(xgboost)
library(dplyr)
library(ggplot2)
library(corrplot)

# ============================================================================ #
# Step 1: Load and inspect the dataset
# ============================================================================ #

load("C:/Users/u229860/Documents/01_Studium/SBD3/Git_1_Assignment_Project/SBD3_1_Assignment/Carlos_Test/data_wage.RData")

dim(data)          # 10,809 observations, 78 variables
summary(data$wage) # Overview of wage distribution

# ============================================================================ #
# Step 2: Data preparation – understand and filter the target variable
# ============================================================================ #

cat("Observations with wage  = 0:", sum(data$wage == 0), "\n")
cat("Observations with wage  > 0:", sum(data$wage > 0),  "\n")

# The 1,000 zero-wage entries represent students or non-respondents.
# Including them would bias the model (predicting 0 for non-earners is a
# different problem than predicting actual salary levels).
# We therefore filter to only observations with a reported, positive salary.
data_model <- data |> filter(wage > 0)

cat("Final modeling dataset:", nrow(data_model), "observations\n")

# Visualize the distribution of the target variable
ggplot(data_model, aes(x = wage)) +
  geom_histogram(bins = 60, fill = "steelblue", color = "white") +
  scale_x_continuous(labels = scales::comma) +
  labs(title = "Distribution of Wages (Filtered: wage > 0)",
       x = "Yearly Wage (USD)", y = "Count")

# ============================================================================ #
# Step 3: Feature matrix preparation (One-Hot Encoding)
# ============================================================================ #

# XGBoost requires a fully numeric matrix. model.matrix() handles this by:
# 1. Converting factor variables (e.g. "country") into dummy columns (one-hot encoding)
# 2. Keeping numeric columns (0/1 indicators) unchanged
# 3. The intercept (-1) is omitted since XGBoost does not need it
# The target variable "wage" is excluded from the predictor matrix.

feature_matrix <- model.matrix(~ . - wage - 1, data = data_model)

cat("Feature matrix:", nrow(feature_matrix), "rows x", ncol(feature_matrix), "columns\n")

# ============================================================================ #
# Step 4: Train / Test Split (70% / 30%)
# ============================================================================ #

# A proper ML pipeline requires holding out a test set that the model
# NEVER sees during training. We use 70% for training, 30% for evaluation.
set.seed(4821)

split     <- initial_split(data_model, prop = 0.7)
train_idx <- split$in_id
test_idx  <- setdiff(seq_len(nrow(data_model)), train_idx)

x_train <- feature_matrix[train_idx, ]
x_test  <- feature_matrix[test_idx,  ]
y_train <- data_model$wage[train_idx]
y_test  <- data_model$wage[test_idx]

cat("Training set:", length(y_train), "observations\n")
cat("Test set:    ", length(y_test),  "observations\n")

# ============================================================================ #
# Step 5: Create DMatrix (XGBoost-internal data format)
# ============================================================================ #

dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest  <- xgb.DMatrix(data = x_test,  label = y_test)

# ============================================================================ #
# Step 6: Define XGBoost hyperparameters
# ============================================================================ #
set.seed(7)

# ============================================================================ #
# Used in Lecture
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
# ============================================================================ #

params <- list(
  objective        = "reg:squarederror",
  eval_metric      = "rmse",
  eta              = 0.05,
  max_depth        = 6,
  min_child_weight = 5,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  gamma            = 0.1
)

# ============================================================================ #
# Step 7: Train the XGBoost model
# ============================================================================ #

set.seed(4821)

xgb_model <- xgb.train(
  params                = params,
  data                  = dtrain,
  nrounds               = 1000,            # Max. number of boosting iterations (trees)
  evals                 = list(train = dtrain, test = dtest),
  early_stopping_rounds = 30,              # Stop if test RMSE doesn't improve for 30 rounds
  verbose               = 1               # Print RMSE progress to console
)

# ============================================================================ #
# Step 8: Make predictions on the test set
# ============================================================================ #

pred_wage <- predict(xgb_model, newdata = dtest)

# Quick sanity check: compare first 10 predicted vs actual wages
data.frame(
  Actual_Wage    = round(y_test[1:10]),
  Predicted_Wage = round(pred_wage[1:10])
)

# ============================================================================ #
# Step 9: Model evaluation – regression performance metrics
# ============================================================================ #

# RMSE – Root Mean Squared Error
# Average prediction error in USD. Larger errors are penalized more heavily.
rmse <- sqrt(mean((pred_wage - y_test)^2))

# MAE – Mean Absolute Error
# Average absolute deviation in USD. More robust to outliers than RMSE.
mae  <- mean(abs(pred_wage - y_test))

# R² – Coefficient of Determination
# Proportion of wage variance explained by the model.
# 0 = model explains nothing, 1 = perfect prediction
ss_res <- sum((pred_wage - y_test)^2)
ss_tot <- sum((y_test - mean(y_test))^2)
r2     <- 1 - (ss_res / ss_tot)

cat("\n========== XGBoost Model Performance (Test Set) ==========\n")
cat("RMSE: $", format(round(rmse), big.mark = ","), "USD\n")
cat("MAE:  $", format(round(mae),  big.mark = ","), "USD\n")
cat("R²:      ", round(r2, 4), "\n")
cat("===========================================================\n")

# Predicted vs. Actual scatter plot
ggplot(data.frame(actual = y_test, predicted = pred_wage),
       aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.2, color = "steelblue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linewidth = 1) +
  scale_x_continuous(labels = scales::comma) +
  scale_y_continuous(labels = scales::comma) +
  labs(title = "XGBoost: Predicted vs. Actual Wages",
       subtitle = paste0("R² = ", round(r2, 3),
                         "  |  RMSE = $", format(round(rmse), big.mark = ",")),
       x = "Actual Wage (USD)", y = "Predicted Wage (USD)")

