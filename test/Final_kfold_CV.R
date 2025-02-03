#Math 7636/8636 - Adv. Stat Learning II
#Group 9
#RF-CV model

#Data link: https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption

# Clear the R workspace/environment
rm(list = ls())

#load data
data=read.csv("Steel_industry_data.csv")
dim(data)
head(data,2)
#display all column names
names(data)

#exclude load_type feature as it is similar to response
data1 = data[, c("date","Usage_kWh" ,"Lagging_Current_Reactive.Power_kVarh", "Leading_Current_Reactive_Power_kVarh", 
                  "CO2.tCO2.", "Lagging_Current_Power_Factor", "Leading_Current_Power_Factor" , "NSM", "WeekStatus", 
                  "Day_of_week")]

dim(data1)  # Display the dimensions (number of rows and columns) of the dataset.
sum(is.na(data1))# Count the number of missing values in the "Salary" column.

str(data1)

#check unique class for each categorical features
unique(data1$WeekStatus)
unique(data1$Day_of_week)

# Convert categorical features  to numerical.                         
data1$WeekStatus=as.numeric(factor(data1$WeekStatus))
data1$Day_of_week=as.numeric(factor(data1$Day_of_week))

#check unique class for each categorical features after converting to numerical
unique(data1$WeekStatus)
unique(data1$Day_of_week)

# Create new varaible/feature "Month" based on "Date"
# Load the lubridate package for date-time manipulation
library(lubridate)
# Convert the "date" column to a DateTime object
data1$date = dmy_hm(data1$date)  # Assumes "date" column contains character date-time values

# Extract the month and store it as a numeric variable
Month = month(data1$date)
unique(Month)

# Remove the date feature
data1$date = NULL

# Combine the original data with the Month
data2 = cbind(data1, Month)
names(data2)

head(data2)

# Compute correlation matrix
correlation_matrix <- cor(data2)

# Display correlation values with Usage_kWh 
correlation_with_Usage_kWh <- correlation_matrix["Usage_kWh", ]
print(correlation_with_Usage_kWh )

#par(mfrow = c(1, 3))

#pairs(data)

# Example scatter plot for "Usage_kWh" against "Lagging_Current_Reactive.Power"
plot(data2$"Lagging_Current_Reactive.Power_kVarh", data2$Usage_kWh, 
     xlab = "Lagging_Current_Reactive.Power", ylab = "Usage_kWh", 
     main = "Energy Consumption vs. Lagging Current Reactive Power")

plot(data2$"Leading_Current_Reactive_Power_kVarh", data2$Usage_kWh, 
     xlab = "Leading_Current_Reactive_Power", ylab = "Usage_kWh", 
     main = "Energy Consumption vs. Leading Current Reactive_Power")

plot(data2$"CO2.tCO2.", data2$Usage_kWh, 
     xlab = "CO2.tCO2.", ylab = "Usage_kWh", 
     main = "Energy Consumption vs. CO2 emmision")

plot(data2$"Lagging_Current_Power_Factor", data2$Usage_kWh, 
     xlab = " Lagging_Current_Power_Factor", ylab = "Usage_kWh", 
     main = "Energy Consumption vs.  Lagging_Current_Power_Factor")

plot(data2$"Leading_Current_Power_Factor", data2$Usage_kWh, 
     xlab = " Leading_Current_Power_Factor", ylab = "Usage_kWh", 
     main = "Energy Consumption vs.  Leading_Current_Power_Factor")

plot(data2$NSM, data2$Usage_kWh, 
     xlab = "NSM", ylab = "Usage_kWh", 
     main = "Energy Consumption vs. NSM")

plot(data2$WeekStatus , data2$Usage_kWh, 
     xlab = "WeekStatus", ylab = "Usage_kWh", 
     main = "Energy Consumption vs. WeekStatus")

plot(data2$Day_of_week, data2$Usage_kWh, 
     xlab = "Day_of_week", ylab = "Usage_kWh", 
     main = "Energy Consumption vs. Day_of_week")

plot(data2$Month, data2$Usage_kWh, 
     xlab = "Month", ylab = "Usage_kWh", 
     main = "Energy Consumption vs. Month")

# Calculate the correlation between each feature and "Usage_kWh"
correlation_results <- sapply(data2[, -1], function(x) cor(x, data2$Usage_kWh))

# Display the correlation results
print(correlation_results)

library(corrplot)
corrplot(cor(data2), method="color")

names(data2)
str(data2)

set.seed(1)
# Create a 70%-30% train-test split
test = sample(nrow(data2), 0.3 * nrow(data2))
# Specify train data
train = (1:nrow(data2))[-test] #Select rest of the data i.e 80% other than test data as training data
train_data <- data2[train, ]
names(train_data)
test_data <- data2[test, ]

train_output <- train_data$Usage_kWh
test_output <- test_data$Usage_kWh


library(caret)
library(randomForest)

# Define the control parameters for 5-fold cross-validation
set.seed(123)
ctrl <- trainControl(method = "cv", number = 5)

# Train the Random Forest model using 5-fold cross-validation on the train_data
rf_model_cv <- train(Usage_kWh ~., data = train_data, method = "rf", trControl = ctrl)

# View the model details and performance from cross-validation
print(rf_model_cv)


# Make predictions on the test_data using the cross-validated model
predictions_cv <- predict(rf_model_cv, newdata = test_data)

# Calculate Mean Squared Error (MSE) on test data using cross-validated model
mse_cv <- mean((test_output - predictions_cv)^2)
mse_cv

# Calculate R-squared on test data using cross-validated model
r_squared_cv <- 1 - sum((test_output - predictions_cv)^2) / sum((test_output - mean(test_output))^2)

# Print the metrics
cat("MSE for RF-Cv model on test data:", mse_cv, "\n")
cat("R-squared for RF-Cv model on test data on test data:", r_squared_cv, "\n")


