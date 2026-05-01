# ============================================================
# Time Series Forecasting in R
# Models: MLP, Simple RNN, LSTM
# Dataset: Tesla stock (TSLA)
# Target: next-day log return
# ============================================================

# -----------------------------
# 1. Libraries
# -----------------------------
library(tidyverse)
library(lubridate)
library(keras3)
library(tensorflow)
library(tidyquant)
library(zoo)

theme_set(theme_minimal())

# -----------------------------
# 2. Reproducibility
# -----------------------------
set.seed(123)
tensorflow::tf$random$set_seed(123)

# -----------------------------
# 3. Load data
# -----------------------------
df_raw <- tq_get("TSLA", from = "2015-01-01", to = Sys.Date())

df <- df_raw %>%
  select(date, adjusted, volume, high, low, open, close) %>%
  rename(timestamp = date) %>%
  mutate(timestamp = as.POSIXct(timestamp, tz = "UTC")) %>%
  arrange(timestamp)

cat("Failed timestamps:", sum(is.na(df$timestamp)), "\n")
glimpse(df)
dim(df)

# -----------------------------
# 4. Feature engineering
# Predict next-day log return using past sequential features

# Variables we create:
# log_return — The logarithmic daily return measuring the proportional change in Tesla’s price from the previous day.
# volume_log — The logarithm of trading volume used to reduce skewness and scale large volume values.
# volume_change — The log change in trading volume from the previous day capturing shifts in market activity.
# hl_range — The logarithmic ratio of the daily high to low price representing the intraday trading range (a volatility proxy).
# oc_return — The log return from market open to close measuring the intraday price movement.
# ma_5 — The 5-day moving average of the adjusted price used to capture the short-term trend.
# ma_20 — The 20-day moving average of the adjusted price representing a medium-term trend.
# ma_ratio_5 — The percentage deviation of the current price from its 5-day moving average indicating short-term momentum.
# ma_ratio_20 — The percentage deviation of the current price from its 20-day moving average indicating medium-term momentum.
# vol_5 — The 5-day rolling standard deviation of log returns measuring short-term volatility.
# vol_20 — The 20-day rolling standard deviation of log returns measuring longer-term volatility.
# momentum_5 — The 5-day price momentum measuring the percentage change in price relative to 5 days ago.
# momentum_20 — The 20-day price momentum measuring the percentage change in price relative to 20 days ago.

# -----------------------------
df <- df %>%
  arrange(timestamp) %>%
  mutate(
    log_return = log(adjusted / lag(adjusted)),
    volume_log = log(volume),
    volume_change = log(volume / lag(volume)),
    hl_range = log(high / low),
    oc_return = log(close / open),
    ma_5 = zoo::rollmean(adjusted, k = 5, fill = NA, align = "right"),
    ma_20 = zoo::rollmean(adjusted, k = 20, fill = NA, align = "right"),
    ma_ratio_5 = adjusted / ma_5 - 1,
    ma_ratio_20 = adjusted / ma_20 - 1,
    vol_5 = zoo::rollapply(log_return, width = 5, FUN = sd, fill = NA, align = "right"),
    vol_20 = zoo::rollapply(log_return, width = 20, FUN = sd, fill = NA, align = "right"),
    momentum_5 = adjusted / lag(adjusted, 5) - 1,
    momentum_20 = adjusted / lag(adjusted, 20) - 1
  ) %>%
  drop_na()

# -----------------------------
# 5. Quick exploration
# -----------------------------
summary(df)

ggplot(df, aes(x = timestamp, y = log_return)) +
  geom_line() +
  labs(title = "Tesla daily log returns", x = NULL, y = "Log return")

ggplot(df, aes(x = timestamp, y = vol_20)) +
  geom_line() +
  labs(title = "Tesla 20-day rolling volatility", x = NULL, y = "Volatility")

# -----------------------------
# 6. Train-test split
# -----------------------------
train_size <- floor(nrow(df) * 0.9)

train <- df[1:train_size, ]
test  <- df[(train_size + 1):nrow(df), ]

cat("Train size:", nrow(train), "\n")
cat("Test size:", nrow(test), "\n")

# -----------------------------
# 7. Robust scaling

# Scaling is needed because neural networks train much more efficiently 
# and stably when input values are on a similar numerical scale.

# Scaling makes the optimization process numerically stable and helps the neural 
# network learn patterns more efficiently.

# Only scale variables used for modelling
# -----------------------------
model_features <- c(
  "log_return",
  "volume_change",
  "hl_range",
  "oc_return",
  "ma_ratio_5",
  "ma_ratio_20",
  "vol_5",
  "vol_20",
  "momentum_5",
  "momentum_20"
)

target_col <- "log_return"

robust_fit <- function(x) {
  med <- median(x, na.rm = TRUE)
  iqr_val <- IQR(x, na.rm = TRUE)
  if (iqr_val == 0 || is.na(iqr_val)) iqr_val <- 1
  list(median = med, iqr = iqr_val)
}

robust_transform <- function(x, fit) {
  (x - fit$median) / fit$iqr
}

inverse_transform <- function(x_scaled, scaler) {
  x_scaled * scaler$iqr + scaler$median
}

feature_scalers <- lapply(train[model_features], robust_fit)
names(feature_scalers) <- model_features

for (col in model_features) {
  train[[col]] <- robust_transform(train[[col]], feature_scalers[[col]])
  test[[col]]  <- robust_transform(test[[col]], feature_scalers[[col]])
}

target_scaler <- feature_scalers[[target_col]]

cat("Train shape:", dim(train), "\n")
cat("Test shape:", dim(test), "\n")
cat("Missing values in train after scaling:", sum(is.na(train[model_features])), "\n")
cat("Missing values in test after scaling:", sum(is.na(test[model_features])), "\n")

# -----------------------------
# 8. Create sequence dataset
# Correct 3D array shape: samples x time_steps x features

# This function converts the time series into a supervised learning dataset by 
# creating rolling windows of the previous 60 days of inputs to predict 
# the next day’s price (y), formatted as a 3-dimensional array required by neural 
# network models like RNNs and LSTMs.
# (samples, time steps, features)
# -----------------------------
create_dataset <- function(data, feature_cols, target_col, time_steps = 60) {
  X_list <- list()
  y_vec <- c()
  
  feature_data <- data %>% select(all_of(feature_cols))
  y_data <- data[[target_col]]
  n <- nrow(feature_data)
  
  for (i in 1:(n - time_steps)) {
    X_list[[i]] <- as.matrix(feature_data[i:(i + time_steps - 1), , drop = FALSE])
    y_vec[i] <- y_data[i + time_steps]
  }
  
  X_array <- simplify2array(X_list)      # time_steps x features x samples
  X_array <- aperm(X_array, c(3, 1, 2))  # samples x time_steps x features
  
  list(X = X_array, y = as.numeric(y_vec))
}

# -----------------------------
# 9. Sequence length
# Longer window gives RNN/LSTM more temporal context
# -----------------------------
time_steps <- 60

train_seq <- create_dataset(
  data = train,
  feature_cols = model_features,
  target_col = target_col,
  time_steps = time_steps
)

# 2455 sequences (samples); 60 time steps per sequence; 10 features per time step
dim(train_seq$X)

# Example 1 sequence
# rows - time steps 
# columns - features
train_seq$X[1,,]

# 1 sequence over time step 1
train_seq$X[1,1,]

# Feature 3 over 1 sequence 
train_seq$X[1,,3]

# Important: include tail(train, time_steps) so first test windows
# can use the last training observations
test_input <- bind_rows(tail(train, time_steps), test)

test_seq_all <- create_dataset(
  data = test_input,
  feature_cols = model_features,
  target_col = target_col,
  time_steps = time_steps
)

# Keep only sequences whose targets belong to the real test period
X_train <- train_seq$X
y_train <- train_seq$y

X_test <- test_seq_all$X
y_test <- test_seq_all$y

cat("X_train shape:", paste(dim(X_train), collapse = " x "), "\n")
cat("X_test shape:", paste(dim(X_test), collapse = " x "), "\n")
cat("y_train shape:", length(y_train), "\n")
cat("y_test shape:", length(y_test), "\n")
cat("Non-finite values in X_train:", sum(!is.finite(X_train)), "\n")
cat("Non-finite values in X_test:", sum(!is.finite(X_test)), "\n")

# -----------------------------
# 10. Explicit validation split
# We will use the last 10% of the training sequences as a validation set to 
# monitor for overfitting during training. We also need to make sure that 
# time ordering is perserved
# -----------------------------
val_fraction <- 0.10
n_train_seq <- dim(X_train)[1]
val_size <- floor(n_train_seq * val_fraction)
train_end <- n_train_seq - val_size

X_train_sub <- X_train[1:train_end, , , drop = FALSE]
y_train_sub <- y_train[1:train_end]

X_val <- X_train[(train_end + 1):n_train_seq, , , drop = FALSE]
y_val <- y_train[(train_end + 1):n_train_seq]

cat("Training sequences:", dim(X_train_sub)[1], "\n")
cat("Validation sequences:", dim(X_val)[1], "\n")
cat("Test sequences:", dim(X_test)[1], "\n")

# -----------------------------
# 11. Metrics helper
# -----------------------------
# We will evaluate the models using multiple metrics to get a comprehensive 
# view of their performance:
# explained_variance — Measures the proportion of variance in the true values
# that is captured by the predictions, indicating how well the model explains 
# the variability in the data.
explained_variance <- function(y_true, y_pred) {
  if (var(y_true) == 0) return(NA_real_)
  1 - var(y_true - y_pred) / var(y_true)
}

# max_error_metric — Computes the maximum absolute error between the true and 
# predicted values
max_error_metric <- function(y_true, y_pred) {
  max(abs(y_true - y_pred))
}

# directional_accuracy — Calculates the percentage of predictions that correctly 
# predict the direction of change (up or down) in the target variable, which is
# particularly relevant for financial forecasting.
directional_accuracy <- function(y_true, y_pred) {
  mean(sign(y_true) == sign(y_pred))
}

# evaluate_model — A comprehensive function that computes multiple evaluation metrics
# for the true and predicted values.
evaluate_model <- function(y_true, y_pred) {
  keep <- is.finite(y_true) & is.finite(y_pred)
  
  if (sum(keep) == 0) {
    return(tibble(
      RMSE = NA_real_,
      MAE = NA_real_,
      MSE = NA_real_,
      Explained_Variance = NA_real_,
      Max_Error = NA_real_,
      Directional_Accuracy = NA_real_
    ))
  }
  
  y_true <- y_true[keep]
  y_pred <- y_pred[keep]
  
  tibble(
    RMSE = sqrt(mean((y_true - y_pred)^2)),
    MAE = mean(abs(y_true - y_pred)),
    MSE = mean((y_true - y_pred)^2),
    Explained_Variance = explained_variance(y_true, y_pred),
    Max_Error = max_error_metric(y_true, y_pred),
    Directional_Accuracy = directional_accuracy(y_true, y_pred)
  )
}

# plot_history — A function to visualize the training and validation loss over epochs, 
# which helps in diagnosing issues like overfitting or underfitting during model training.
plot_history <- function(history_obj, title_text) {
  hist_df <- tibble(
    epoch = seq_along(history_obj$metrics$loss),
    loss = history_obj$metrics$loss,
    val_loss = history_obj$metrics$val_loss
  )
  
  ggplot(hist_df, aes(x = epoch)) +
    geom_line(aes(y = loss, color = "train")) +
    geom_line(aes(y = val_loss, color = "validation")) +
    labs(title = title_text, x = "Epoch", y = "Loss", color = NULL)
}

# plot_predictions — A function to visualize the true vs predicted values for the test set, 
# which provides insights into how well the model is capturing the patterns in the data and can 
# help identify any systematic errors or biases in the predictions.
plot_predictions <- function(y_true_scaled, y_pred_scaled, scaler, title_text) {
  y_true_inv <- inverse_transform(y_true_scaled, scaler)
  y_pred_inv <- inverse_transform(y_pred_scaled, scaler)
  
  plot_df <- tibble(
    idx = seq_along(y_true_inv),
    true = y_true_inv,
    predicted = y_pred_inv
  ) %>%
    filter(is.finite(true), is.finite(predicted))
  
  ggplot(plot_df, aes(x = idx)) +
    geom_line(aes(y = true, color = "true")) +
    geom_line(aes(y = predicted, color = "predicted")) +
    labs(title = title_text, x = "Observation", y = "Next-day log return", color = NULL)
}

# -----------------------------
# 12. Callbacks
# -----------------------------
# Training stops if validation loss does not improve for 10 consecutive epochs.
callbacks_list <- list(
  callback_early_stopping(
    monitor = "val_loss",
    patience = 10,
    restore_best_weights = TRUE
  )
)

# ============================================================
# 13. MLP
# Flatten sequence into one long vector
# You reshape the data because an MLP requires 2-dimensional input (samples × features), 
# so the original 3-dimensional sequence data (samples × time steps × features) is flattened 
# into a single feature vector per sample (samples × time_steps × features).
# ============================================================
X_train_mlp <- array_reshape(
  X_train_sub,
  c(dim(X_train_sub)[1], dim(X_train_sub)[2] * dim(X_train_sub)[3])
)

X_val_mlp <- array_reshape(
  X_val,
  c(dim(X_val)[1], dim(X_val)[2] * dim(X_val)[3])
)

X_test_mlp <- array_reshape(
  X_test,
  c(dim(X_test)[1], dim(X_test)[2] * dim(X_test)[3])
)

cat("X_train_mlp shape:", dim(X_train_mlp), "\n")
cat("X_val_mlp shape:", dim(X_val_mlp), "\n")
cat("X_test_mlp shape:", dim(X_test_mlp), "\n")

# The MLP architecture consists of:
# - An input layer that takes the flattened sequence data.
# - A hidden layer with 64 neurons and ReLU activation to capture complex patterns.
# - A dropout layer with a rate of 0.2 to prevent overfitting by
#  randomly dropping 20% of the neurons during training.
# - A second hidden layer with 32 neurons and ReLU activation for further feature extraction.
# - An output layer with a single neuron to predict the next-day log return.
mlp_model <- keras_model_sequential() %>%
  layer_dense(
    units = 64,
    activation = "relu",
    input_shape = c(ncol(X_train_mlp))
  ) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1)

# The optimizer determines how the model updates its weights during training.
# Here we use Adam (Adaptive Moment Estimation)
mlp_model %>% compile(
  loss = "mse",
  optimizer = optimizer_adam(learning_rate = 0.0005)
)

summary(mlp_model)

# epoch means one full pass through the entire training dataset, while batch_size 
# refers to the number of samples processed before the model's internal parameters are updated. 
# In this case, the model will update its weights after processing every 32 samples, 
# and it will go through the entire training dataset up to 60 times (epochs) or until 
# early stopping is triggered based on validation loss.
mlp_history <- mlp_model %>% fit(
  x = X_train_mlp,
  y = y_train_sub,
  validation_data = list(X_val_mlp, y_val),
  epochs = 60,
  batch_size = 32,
  shuffle = FALSE,
  callbacks = callbacks_list,
  verbose = 1,
  view_metrics = FALSE
)

plot_history(mlp_history, "MLP Training History")

mlp_train_pred <- as.numeric(predict(mlp_model, X_train_mlp))
mlp_val_pred   <- as.numeric(predict(mlp_model, X_val_mlp))
mlp_test_pred  <- as.numeric(predict(mlp_model, X_test_mlp))

plot_predictions(y_test, mlp_test_pred, target_scaler, "MLP: True vs Predicted")

cat("\nMLP metrics\n")
cat("Training\n")
print(evaluate_model(y_train_sub, mlp_train_pred))
cat("Validation\n")
print(evaluate_model(y_val, mlp_val_pred))
cat("Test\n")
print(evaluate_model(y_test, mlp_test_pred))

mean(y_train_sub)

# ============================================================
# 14. Simple RNN
# ============================================================
# We are next building a Simple RNN with 24 recurrent neurons that processes the past 
# 60 days of Tesla market features sequentially and outputs a prediction of the 
# next-day log return using mean squared error optimization with the Adam algorithm.
rnn_model <- keras_model_sequential() %>%
  layer_simple_rnn(
    units = 24,
    input_shape = c(dim(X_train_sub)[2], dim(X_train_sub)[3]),
    dropout = 0.1
  ) %>%
  layer_dense(units = 1)

rnn_model %>% compile(
  loss = "mse",
  optimizer = optimizer_adam(learning_rate = 0.0005)
)

# 10 features * 24 neurons = 240 parameters in the RNN layer + 24 bias terms = 264 parameters
# Recurren weights: 24 neurons * 24 neurons = 576 parameters
# Total parameters in RNN layer = 264 + 576 = 840
summary(rnn_model)

rnn_history <- rnn_model %>% fit(
  x = X_train_sub,
  y = y_train_sub,
  epochs = 60,
  batch_size = 32,
  validation_data = list(X_val, y_val),
  shuffle = FALSE,
  callbacks = callbacks_list,
  verbose = 1,
  view_metrics = FALSE
)

plot_history(rnn_history, "Simple RNN Training History")

rnn_train_pred <- as.numeric(predict(rnn_model, X_train_sub))
rnn_val_pred   <- as.numeric(predict(rnn_model, X_val))
rnn_test_pred  <- as.numeric(predict(rnn_model, X_test))

plot_predictions(y_test, rnn_test_pred, target_scaler, "Simple RNN: True vs Predicted")

cat("\nSimple RNN metrics\n")
cat("Training\n")
print(evaluate_model(y_train_sub, rnn_train_pred))
cat("Validation\n")
print(evaluate_model(y_val, rnn_val_pred))
cat("Test\n")
print(evaluate_model(y_test, rnn_test_pred))

# ============================================================
# 15. LSTM
# ============================================================
lstm_model <- keras_model_sequential() %>%
  layer_lstm(
    units = 32,
    input_shape = c(dim(X_train_sub)[2], dim(X_train_sub)[3]),
    dropout = 0.1
  ) %>%
  layer_dense(units = 1)

lstm_model %>% compile(
  loss = "mse",
  optimizer = optimizer_adam(learning_rate = 0.0005)
)

# LSTM layer parameters:
# Input weights: 10 features * 32 neurons = 320 parameters
# Recurrent weights: 32 neurons * 32 neurons = 1024 parameters
# Biases: 32 neurons
# Gates: 4 * (320 + 1024 + 32) = 4 * 1376 = 5504 parameters in the LSTM layer
summary(lstm_model)

lstm_history <- lstm_model %>% fit(
  x = X_train_sub,
  y = y_train_sub,
  epochs = 60,
  batch_size = 32,
  validation_data = list(X_val, y_val),
  shuffle = FALSE,
  callbacks = callbacks_list,
  verbose = 1,
  view_metrics = FALSE
)

plot_history(lstm_history, "LSTM Training History")

lstm_train_pred <- as.numeric(predict(lstm_model, X_train_sub))
lstm_val_pred   <- as.numeric(predict(lstm_model, X_val))
lstm_test_pred  <- as.numeric(predict(lstm_model, X_test))

plot_predictions(y_test, lstm_test_pred, target_scaler, "LSTM: True vs Predicted")

cat("\nLSTM metrics\n")
cat("Training\n")
print(evaluate_model(y_train_sub, lstm_train_pred))
cat("Validation\n")
print(evaluate_model(y_val, lstm_val_pred))
cat("Test\n")
print(evaluate_model(y_test, lstm_test_pred))

# ============================================================
# 16. Compare models on test set
# ============================================================
results <- bind_rows(
  evaluate_model(y_test, mlp_test_pred)  %>% mutate(Model = "MLP"),
  evaluate_model(y_test, rnn_test_pred)  %>% mutate(Model = "Simple RNN"),
  evaluate_model(y_test, lstm_test_pred) %>% mutate(Model = "LSTM")
) %>%
  select(Model, everything()) %>%
  arrange(RMSE)

print(results)

param_table <- tibble(
  Model = c("MLP", "Simple RNN", "LSTM"),
  Trainable_Parameters = c(
    mlp_model$count_params(),
    rnn_model$count_params(),
    lstm_model$count_params()
  )
)

final_summary <- results %>%
  left_join(param_table, by = "Model") %>%
  select(
    Model,
    Trainable_Parameters,
    RMSE,
    MAE,
    MSE,
    Explained_Variance,
    Directional_Accuracy,
    Max_Error
  )

print(final_summary)
