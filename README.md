Power Consumption Prediction for Steel Industry Using Machine Learning Algorithms:

Objectives:
This project aims to develop a predictive model for estimating energy consumption in the steel industry using machine learning. The model will consider input factors such as reactive power, power factors, and carbon dioxide emissions for various load types. The primary objectives are to identify the key factors influencing energy consumption, enhance understanding of energy usage patterns, and empower the steel industry to optimize energy usage and reduce costs.

Background:
Since 1981, the U.S. Department of Energy Industrial Assessments Centers (IACs) have conducted audits to monitor and analyze energy consumption data within major industries. These assessments examine energy usage across diverse industrial sectors and offer recommendations to enhance energy efficiency. Accurate forecasting of energy consumption is crucial for efficiently managing and improving energy utilization in industries like the steel industry.

Dataset Overview:
The dataset used is "Steel Industry Energy Consumption," gathered by DAEWOO Steel Co. Ltd in Gwangyang, South Korea. It is available on the UC Irvine Machine Learning Repository website. The dataset contains 35,040 observations with 10 features, including date, lagging/leading reactive power, lagging/leading power factor, CO2, week status, and day of the week.

Procedures/Methods
Data Preprocessing:
The following preprocessing steps were applied:

Remove null values: The dataset was checked for null values, and none were found.
Feature extraction: The month was extracted from the date feature.
Remove duplicate features: The "Load_Type" feature was excluded as it was similar to the response variable "Usage_kWh." The "Date" feature was also excluded after extracting the month.
Convert categorical features: Categorical features were converted to numerical ones.
Train-test split: The dataset was divided into a 70% training set and a 30% test set.
Data Visualization
Correlation analysis and scatterplots were used to explore relationships between features and the response variable.

Evaluation Methods
The following evaluation methods were used:

k-fold Cross-Validation: 5-fold cross-validation was used to assess model performance on different data subsets.
Coefficient of determination (R²): R² was used to measure the model's accuracy in predicting energy consumption.
Mean Square Error (MSE): MSE was used to quantify the average squared difference between predicted and actual values.
Statistical Learning Methods
Various statistical learning methods were employed, including:

Multiple Linear Regression Model
Subset Selection
LASSO Regression Model
Ridge Regression Model
Principal Component Analysis (PCA)
Random Forest Model
Random Forest Model with CV
Linear Regression Model with polynomial features
Random Forest Model with polynomial features
Results and Discussion
The results of each model were evaluated based on MSE and R-squared. The Random Forest model with 5-fold Cross-Validation outperformed the other models, achieving an MSE of 1.1488 and an R-squared of 0.999.

Conclusion
The project successfully explored various statistical learning methods for predicting energy consumption in the steel industry. The Random Forest model with 5-fold Cross-Validation emerged as the most accurate model. This model can potentially lead to significant energy savings and cost reductions in the steel industry.

References
S. A. Sarswatula, T. Pugh, and V. Prabhu, "Modeling Energy Consumption Using Machine Learning," Front. Manuf. Technol., vol. 2, 2022.

M. Bahij et al., "A Review on the Prediction of Energy Consumption in the Industry Sector Based on Machine Learning Approaches," in 2021 4th International Symposium on Advanced Electrical and Communication Technologies (ISAECT), Dec. 2021, pp. 01-05.

C. S. Sathishkumar V E, "Steel Industry Energy Consumption." UCI Machine Learning Repository, 2021.

R. Patro, "Cross Validation: K Fold vs Monte Carlo," Medium.

P. Sharma, "Different Types of Regression Models," Analytics Vidhya.

"Random Forest Regression in Python," GeeksforGeeks.