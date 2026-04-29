############################################################
# 00_setup_python_once.R
# Run this ONCE before the practical
# With this we are setting up a working Python deep-learning environment 
# from inside R so that you can run TensorFlow/Keras models through R.

# reticulate → bridge between R and Python

# tensorflow → R interface to TensorFlow

# keras3 → modern R interface to Keras

############################################################

# Step 1: install required R packages if missing
needed_packages <- c(
  "reticulate",
  "tensorflow",
  "keras3",
  "readr",
  "dplyr",
  "caret",
  "DescTools",
  "rsample",
  "pROC",
  "ROCR",
  "randomForest",
  "rpart",
  "rpart.plot",
  "dlookr",
  "xgboost"
)

installed <- rownames(installed.packages())
to_install <- setdiff(needed_packages, installed)

if (length(to_install) > 0) {
  install.packages(to_install)
}

# Step 2: load reticulate and declare Python requirements
library(reticulate)

# This tells reticulate which Python packages are needed.
# reticulate can create and manage an isolated environment automatically.
py_require(
  packages = c("tensorflow", "keras", "numpy"),
  python_version = ">=3.10,<3.13"
)

# Step 3: show Python configuration
py_config()

# Step 4: check that Python modules are available
cat("\nTensorFlow available in Python: ", py_module_available("tensorflow"), "\n")
cat("Keras available in Python: ", py_module_available("keras"), "\n")

# Step 5: test TensorFlow from R
library(tensorflow)
tf$constant("TensorFlow is working")

# Step 6: test keras3 from R
library(keras3)

inputs <- keras_input(shape = c(2))
outputs <- inputs %>%
  layer_dense(units = 4, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model <- keras_model(inputs = inputs, outputs = outputs)

model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = "accuracy"
)

cat("\nSUCCESS: Python + TensorFlow + keras3 are working.\n")
