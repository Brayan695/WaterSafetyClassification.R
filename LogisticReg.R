library(dplyr)
library(ggplot2)
library(caret)


groundwater_data <- read.csv("/Users/theruinedking/Documents/Math 4322/groundwater_data.csv")

# Convert 'Classification.1' to binary
groundwater_data$Classification.1 <- ifelse(groundwater_data$Classification.1 == "P.S.", 1, 0)

# Select specific columns and remove missing values
selected_data <- groundwater_data %>%
  select(RSC, SAR, Na, E.C, TDS, HCO3, pH, mandal, village, Classification.1) %>%
  na.omit()


str(selected_data)  # Should show the expected structure
colnames(selected_data)  # Confirm the selected columns

# Convert categorical variables to factors
selected_data$mandal <- as.factor(selected_data$mandal)
selected_data$village <- as.factor(selected_data$village)

# One-hot encode categorical variables except the response variable
dummy_vars <- model.matrix(~ mandal + village, data = selected_data)

numerical_vars <- selected_data[, c("RSC", "SAR", "Na", "E.C", "TDS", "HCO3", "pH")]  # Numerical predictors
response_var <- selected_data$Classification.1  # Response variable
final_data <- cbind(as.data.frame(dummy_vars), numerical_vars, Classification.1 = response_var)  # Combine with dummy variables


# Check that 'final_data' is a data frame
is.data.frame(final_data)  # Should return TRUE

# Check the structure of 'final_data'
str(final_data)

# Partition the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(final_data$Classification.1, p = 0.7, list = FALSE)
trainData <- final_data[trainIndex, ]
testData <- final_data[-trainIndex, ]

# Build a logistic regression model with 'Classification.1' as the response variable
model <- glm(Classification.1 ~ ., data = trainData, family = binomial)

summary(model)

# Make predictions on the test set
predictions <- predict(model, newdata = testData, type = "response")
predictedClass <- ifelse(predictions > 0.5, 1, 0)

# Create a confusion matrix and calculate accuracy
confusionMatrix <- table(Predicted = predictedClass, Actual = testData$Classification.1)
accuracy <- sum(diag(confusionMatrix)) / sum(confusionMatrix)

# Display results
print("Confusion Matrix:")
print(confusionMatrix)
print(paste("Accuracy:", round(accuracy, 4)))