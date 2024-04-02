library(tree)
library(randomForest)
library(caret)

ground_water_quality_2022_post <- read.csv("C:/Users/braya/OneDrive/Desktop/4337 Project/4337_Project/ground_water_quality_2022_post.csv")

ground_water_quality_2022_post = na.omit(ground_water_quality_2022_post)

# Omitting CO3 and Season since they're the same
ground_water_quality_2022_post$CO3 = NULL
ground_water_quality_2022_post$season = NULL

# Selecting the most useful features
gw_df = ground_water_quality_2022_post[,c(23, 21, 16, 9, 10, 11, 8, 3, 4)]

# One-hot-encoding Mandal and Village
dummy = dummyVars(" ~ .", data = gw_df)
dum_df = data.frame(predict(dummy, newdata = gw_df))

# Changing Marginal Safe (MR) to Unsafe (U.S.)
Classification.1 = ground_water_quality_2022_post[,24]
final_gw_df = data.frame(dum_df, Classification.1)
final_gw_df$Classification.1 = gsub("MR", "U.S.", final_gw_df$Classification.1)
final_gw_df$Classification.1 = as.factor(final_gw_df$Classification.1)

conf_mat = list()

set.seed(1)

for (i in 1:10) {

  # Splitting data
  sample = sample(1:nrow(final_gw_df), 0.8 * nrow(final_gw_df))
  training = final_gw_df[sample,]
  testing = final_gw_df[-sample,]
  
  # Random Forest Model
  rf_gw = randomForest(Classification.1 ~., data = training, proximity = T, importance = T)
  
  # Importance Plots
  importance(rf_gw)
  varImpPlot(rf_gw, main = paste('Ground Water Random Forest Split', i, sep = ' '))
  
  test.pred = predict(rf_gw, newdata = testing)
  
  # Confusion Matrix
  conf_mat[[i]] = table(testing$Classification.1, test.pred)
  
}

# Test Error Calculations
test_errors = rep(0, 10)

# Number 1
conf_mat[[1]]
test_errors[1] = sum(0,4)/sum(152,0,4,8)

# Number 2
conf_mat[[2]]
test_errors[2] = sum(0,1)/sum(150,0,1,13)

# Number 3
conf_mat[[3]]
test_errors[3] = sum(0,1)/sum(151,0,1,12)

# Number 4
conf_mat[[4]]
test_errors[4] = sum(0,1)/sum(153,0,1,10)

# Number 5
conf_mat[[5]]
test_errors[5] = sum(0,1)/sum(148,0,1,15)

# Number 6
conf_mat[[6]]
test_errors[6] = sum(0,0)/sum(154,0,0,10)

# Number 7
conf_mat[[7]]
test_errors[7] = sum(0,2)/sum(154,0,2,8)

# Number 8
conf_mat[[8]]
test_errors[8] = sum(0,0)/sum(153,0,0,11)

# Number 9
conf_mat[[9]]
test_errors[9] = sum(0,2)/sum(155,0,2,7)

# Number 10
conf_mat[[10]]
test_errors[10] = sum(0,2)/sum(156,0,2,6)

test_errors

