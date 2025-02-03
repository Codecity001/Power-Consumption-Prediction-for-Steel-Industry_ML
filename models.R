#HW3

#Exploring Steel_industry_data
#Data link: https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption
#load data
s_data=read.csv("E:\\Imp\\MS\\Adv-2\\Project\\Steel_industry_data.csv")
names(s_data)

#exclude load_type feature
data = s_data[, c("date","Usage_kWh" ,"Lagging_Current_Reactive.Power_kVarh", "Leading_Current_Reactive_Power_kVarh", 
                  "CO2.tCO2.", "Lagging_Current_Power_Factor", "Leading_Current_Power_Factor" , "NSM", "WeekStatus", 
                  "Day_of_week")]

dim(data)  # Display the dimensions (number of rows and columns) of the dataset.
sum(is.na(data))# Count the number of missing values in the "Salary" column.

#check unique class for each categorical features
unique(data$WeekStatus)
unique(data$Day_of_week)

# Convert categorical features  to numerical.                         
data$WeekStatus=as.numeric(factor(data$WeekStatus))
data$Day_of_week=as.numeric(factor(data$Day_of_week))

unique(data$WeekStatus)
unique(data$Day_of_week)

# Create new varaible/feature "Month" based on "Date"
# Load the lubridate package for date-time manipulation
library(lubridate)
# Convert the "date" column to a DateTime object
data$date = dmy_hm(data$date)  # Assumes "date" column contains character date-time values

# Extract the month and store it as a numeric variable
Month = month(data$date)
unique(Month)

# Remove the date feature
data$date = NULL

# Combine the original data with the Month
data = cbind(data, Month)
head(data)

par(mfrow = c(1, 3))
plot(data$Month,data$Usage_kWh,xlab="Month",ylab="Energy Consumption")
plot(data$WeekStatus,data$Usage_kWh,xlab="Week",ylab="Energy Consumption")
plot(data$Day_of_week,data$Usage_kWh,xlab="Day",ylab="Energy Consumption")

pairs(data)

# Example scatter plot for "Usage_kWh" against "NSM"
plot(s_data$NSM, s_data$Usage_kWh, 
     xlab = "NSM", ylab = "Usage_kWh", 
     main = "Scatter Plot of Usage_kWh vs. NSM")

plot(s_data$NSM, s_data$Usage_kWh, 
     xlab = "CO2.tCO2.", ylab = "Usage_kWh", 
     main = "Scatter Plot of Usage_kWh vs. CO2.tCO2.")



# Calculate the correlation between each feature and "Usage_kWh"
correlation_results <- sapply(data[, -1], function(x) cor(x, s_data$Usage_kWh))

# Display the correlation results
print(correlation_results)

set.seed(1)
# Create a 70%-30% train-test split
test = sample(nrow(data), 0.3 * nrow(data))
# Specify train data
train = (1:nrow(data))[-test] #Select rest of the data i.e 80% other than test data as training data
train_data <- data[train, ]
test_data <- data[test, ]

names(data)


train_output <- train_data$Usage_kWh
test_output <- test_data$Usage_kWh


##LR MODEL
# Train a Linear Regression model with all featrures
lm_model <- lm(Usage_kWh ~ ., data = train_data)

# Display a summary of the Linear Regression model
summary(lm_model)

# Make predictions on the test data
predictions <- predict(lm_model, newdata = test_data)

# Evaluate the model using metrics (e.g., RMSE, R-squared) to assess the model's performance.
#calculate the MSE:
mse <- (mean((test_output - predictions)^2))
#cat(" Mean Squared Error (MSE):", mse, "\n")

#Calculae R-squared:
r_squared <- 1 - (sum((test_output - predictions)^2) / sum((test_output - mean(test_output))^2))
#cat("R-squared:", r_squared, "\n")

cat(" Mean Squared Error (MSE):", mse, "\n")
cat("R-squared:", r_squared, "\n")


# Specify the degree of the polynomial
degree <- 2

# Extract numeric predictors (excluding the response variable "Usage_kWh" and non-numeric columns)
numeric_predictors <- train_data[, sapply(train_data, is.numeric) & names(train_data) != "Usage_kWh"]

# Ensure all values in numeric predictors are numeric
numeric_predictors <- sapply(numeric_predictors, as.numeric)

# Check for missing values and replace them with 0
numeric_predictors[is.na(numeric_predictors)] <- 0

# Apply polynomial transformation to numeric predictors
poly_features_train <- as.data.frame(poly(numeric_predictors, degree, raw = TRUE))

# Combine the polynomial features with the original training data
train_data_poly <- cbind(train_data, poly_features_train)

# Train a Polynomial Regression model with the polynomial features
poly_model <- lm(Usage_kWh ~ ., data = train_data_poly)

# Display a summary of the Polynomial Regression model
summary(poly_model)

# Apply the same polynomial transformation to the test data
numeric_predictors_test <- test_data[, sapply(test_data, is.numeric) & names(test_data) != "Usage_kWh"]
numeric_predictors_test <- sapply(numeric_predictors_test, as.numeric)
numeric_predictors_test[is.na(numeric_predictors_test)] <- 0

poly_features_test <- as.data.frame(poly(numeric_predictors_test, degree, raw = TRUE))
test_data_poly <- cbind(test_data, poly_features_test)

# Make predictions on the test data using the polynomial model
predictions_poly <- predict(poly_model, newdata = test_data_poly)

# Evaluate the polynomial model using metrics (e.g., RMSE, R-squared)
mse_poly <- mean((test_output - predictions_poly)^2)
r_squared_poly <- 1 - (sum((test_output - predictions_poly)^2) / sum((test_output - mean(test_output))^2))

cat("Polynomial Regression Model (Using All Features):\n")
cat("Mean Squared Error (MSE):", mse_poly, "\n")
cat("R-squared:", r_squared_poly, "\n")



##RF MODEL
# Load the randomForest package (if you haven't already)
library(randomForest)

# Train a Random Forest model with all features
rf_model <- randomForest(Usage_kWh ~ ., data = train_data)

# Print the summary of the Random Forest model (optional)
summary_rf <- summary(rf_model)

# Extract variable importance
variable_importance <- importance(rf_model)

# Print variable importance and node counts
print(variable_importance)

# Set the number of folds for cross-validation
num_folds <- 5

# Create a matrix to store the cross-validated results
cv_results <- matrix(NA, nrow = num_folds, ncol = 2)

# Create a list to store models from each fold
rf_models <- list()

# Perform cross-validation
for (i in 1:num_folds) {
  # Create training and testing datasets for the current fold
  fold_indices <- ((i - 1) * nrow(train_data) / num_folds + 1):(i * nrow(train_data) / num_folds)
  test_data <- train_data[fold_indices, ]
  train_data_fold <- train_data[-fold_indices, ]
  
  # Fit a random forest model on the training data
  rf_model <- randomForest(Usage_kWh ~ ., data = train_data_fold, ntree = 500)
  
  # Save the model
  rf_models[[i]] <- rf_model
  
  # Make predictions on the test data
  predictions <- predict(rf_model, newdata = test_data)
  
  # Calculate the mean squared error (MSE) for the fold
  mse <- mean((test_data$Usage_kWh - predictions)^2)
  
  # Save the results
  cv_results[i, ] <- c(i, mse)
}

# Print cross-validated results
colnames(cv_results) <- c("Fold", "MSE")
print(cv_results)

# Choose the best fold based on the lowest MSE
best_fold <- which.min(cv_results[, "MSE"])

# Use the model from the best fold
final_rf_model <- rf_models[[best_fold]]

# Make predictions on the test dataset using the final model
final_predictions_rf <- predict(final_rf_model, newdata = test_data)


# Calculate the Mean Squared Error (MSE)
mse <- mean((test_data$Usage_kWh - final_predictions_rf)^2)

# Print the final MSE
cat("Final Mean Squared Error (MSE):", mse, "\n")

#subset selection
library(leaps)

# Define the response variable
Y <- train_data$Usage_kWh

# Define the predictor variables, excluding the response
X <- train_data[, -1]

dim(X)
names(X)
# Perform Best Subset Selection
best_subsets <- regsubsets(Y ~ ., data = X, method = "forward")

# Display the summary of the Best Subset Selection
reg.summary1=summary(best_subsets)

#adjr2 (Adusted R-squared)
reg.summary1$adjr2 # Display the Adusted R-squared values for different subsets (9)of predictors.
plot(reg.summary1$adjr2,xlab="Number of Variables",ylab="Adjusted RSq",type="l")
which.max(reg.summary1$adjr2)
points(8,reg.summary1$adjr2[7],col="red",cex=2,pch=20)

## variable model with adjr2
coef(best_subsets, 8)
summary(best_subsets)

#LASSO
# The Lasso

#use cross-validation to choose the tuning parameter λ.
set.seed(1)
library(glmnet)

X <- as.matrix(X)
Y <- as.matrix(Y)

cv.out1 = cv.glmnet(X, Y, alpha = 1)
plot(cv.out1)
bestlam1=cv.out1$lambda.min
bestlam1

#using bestlam1

lassoe.coef1=predict(cv.out1, s = bestlam1, type = "coefficients")[1:9, ]
lassoe.coef1

#Zero Coefficients:
lasso.coef1[lasso.coef1==0]

#Non-Zero Coefficients:
lasso.coef1[lasso.coef1!=0] 


#RIDGE
#use cross-validation to choose the tuning parameter λ.
set.seed(1)
library(glmnet)

cv.out2 = cv.glmnet(X, Y, alpha = 0)
plot(cv.out2)
bestlam2=cv.out2$lambda.min
bestlam2

#using bestlam2
ridge.coef2=predict(cv.out2, s = bestlam2, type = "coefficients")[1:9, ]
ridge.coef2

#Zero Coefficients:
ridge.coef2[ridge.coef2==0]

#Non-Zero Coefficients:
ridge.coef2[ridge.coef2!=0] 

#PCA
# Principal Components Regression and PCA 

x = model.matrix(Usage_kWh~., data = train_data)
pca = prcomp(x, center = T, scale = F)

# Get the first principal component's loading scores
first_pc_loadings <- pca$rotation[, 1]

# Calculate the contribution of each original feature to the first principal component
feature_contributions <- abs(first_pc_loadings)

# Sort the features by their contributions to the first PC in descending order
sorted_features <- names(sort(feature_contributions, decreasing = TRUE))
sorted_features

# Select the top k features (e.g., top 10)
k <- 6
selected_features <- sorted_features[1:k]

# Create a new dataset with only the selected features
selected_data <- train_data[, c("Usage_kWh", selected_features)]

# Perform PCR with selected features
model_pcr <- lm(Usage_kWh ~ ., data = selected_data)

# Print the summary of the PCR model
summary(model_pcr)

# Make predictions on the test data
x_test <- model.matrix(Usage_kWh ~ ., data = test_data)
pca_test <- prcomp(x_test, center = TRUE, scale. = FALSE)
test_data_pca <- pca_test$x[, 1:k]

# Predict Usage_kWh on the test data using the PCR model
predictions <- predict(model_pcr, newdata = as.data.frame(test_data_pca))

# Calculate and print the RMSE (Root Mean Square Error)
mse <- (mean((test_data$Usage_kWh - predictions)^2))
cat("MSE:", mse, "\n")


