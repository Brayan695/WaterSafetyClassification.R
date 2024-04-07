library(tree)
library(randomForest)
library(caret)

# Load Data
ground_water_quality_2022_post <- read.csv("ground_water_quality_2022_post.csv")
ground_water_quality_2022_post <- na.omit(ground_water_quality_2022_post)

# Omitting CO3 and Season since they're the same
ground_water_quality_2022_post$CO3 <- NULL
ground_water_quality_2022_post$season <- NULL

# Selecting the most useful features
gw_df <- ground_water_quality_2022_post[, c(23, 21, 16, 9, 10, 11, 8, 3, 4)]

# One-hot-encoding Mandal and Village
dummy <- dummyVars(" ~ .", data = gw_df)
dum_df <- data.frame(predict(dummy, newdata = gw_df))

# Changing Marginal Safe (MR) to Unsafe (U.S.)
Classification.1 <- ground_water_quality_2022_post[, 24]
final_gw_df <- data.frame(dum_df, Classification.1)
final_gw_df$Classification.1 <- factor(gsub("MR", "U.S.", final_gw_df$Classification.1))

# Ensure no missing values in the dataset
if (anyNA(final_gw_df)) {
  stop("Dataset contains missing values. Please handle missing values before proceeding.")
}

# Define parameter grid for grid search
param_grid <- expand.grid(
  n_estimators = c(50, 100, 200),
  max_depth = c(5, 10, 15),
  min_samples_split = c(2, 5, 10),
  min_samples_leaf = c(1, 3, 5),
  max_features = c("sqrt", "log2"),  # Ensure max_features is specified as characters
  stringsAsFactors = FALSE  # Ensure strings are not converted to factors
)

# Initialize empty list to store results
results <- list()

# Initialize cross-validation folds
folds <- createFolds(final_gw_df$Classification.1, k = 5, list = TRUE, returnTrain = FALSE)

# Perform grid search
for (i in 1:nrow(param_grid)) {
  params <- param_grid[i, ]
  
  # Initialize vector to store cross-validation scores
  cv_scores <- c()
  
  # Perform cross-validation
  for (fold in folds) {
    # Split data into training and testing sets
    training <- final_gw_df[-fold, ]
    testing <- final_gw_df[fold, ]
    
    # Determine the value of mtry based on max_features
    if (params$max_features == "sqrt") {
      mtry_value <- sqrt(ncol(training) - 1)  # Subtract 1 for the target variable
    } else if (params$max_features == "log2") {
      mtry_value <- log2(ncol(training) - 1)
    } else {
      mtry_value <- params$max_features
    }
    
    # Fit random forest model
    rf_gw <- randomForest(Classification.1 ~ ., data = training,
                          ntree = params$n_estimators,
                          mtry = round(mtry_value),  # Round to nearest integer
                          maxdepth = params$max_depth,
                          nodesize = params$min_samples_leaf)
    
    # Make predictions on test set
    test.pred <- predict(rf_gw, newdata = testing)
    
    # Calculate accuracy
    accuracy <- sum(test.pred == testing$Classification.1) / length(testing$Classification.1)
    
    # Append accuracy to vector
    cv_scores <- c(cv_scores, accuracy)
  }
  
  # Store mean cross-validation score in results list
  results[[paste("Model", i, sep = "")]] <- mean(cv_scores)
}

# Print results
print(results)


# Convert results to a data frame for easier manipulation
results_df <- data.frame(Model = names(results), Score = unlist(results))
# Find the row with the highest score
best_model <- results_df[which.max(results_df$Score), ]
# Print the best model
print(best_model)
# Get the name of the best-performing model
best_model_name <- best_model$Model
# Find the corresponding parameters in the param_grid
best_model_params <- param_grid[param_grid$Model == best_model_name, ]
# Print the parameters of the best model
print(best_model_params)


