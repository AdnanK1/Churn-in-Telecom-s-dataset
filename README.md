# Churn Prediction for Enhanced Customer Retention

## Project Overview: Customer Churn Prediction for SyriaTel**

In the dynamic telecommunications industry, retaining a satisfied customer base is crucial. For companies like SyriaTel, tackling customer churn (when customers cease services) is vital for financial stability and growth. This project employs advanced data analysis and predictive modeling to create a powerful classifier that identifies potential churn risks. By leveraging this classifier, SyriaTel aims to proactively address customer churn and enhance its service quality.

## Business Understanding

SyriaTel, a leading telecommunications company, must address customer churn to ensure financial stability and growth. This project aims to create a predictive model identifying potential churn risks, empowering SyriaTel to implement preemptive retention strategies.

### Goals and Objectives

1. **Reduce Service Calls:** Address service quality and billing concerns proactively to decrease customer service calls and mitigate churn risk.

2. **Enhance International Revenue:** Target high international call users with retention incentives, maximizing revenue from international services.

3. **Uncover Churn Patterns:** Visualize churn trends across categories for insights into influencing factors.

4. **Nighttime Service Improvement:** Analyze nighttime call impact on churn, enhance service quality during these hours.

5. **Resource Optimization:** Strategically allocate resources based on churn predictions for efficient operations.

### Problem Statement:

- **Telecom Competition**: Intense competition in the telecom industry challenges customer retention.
- **Customer Churn Challenge**: Companies struggle to retain customers amid enticing offers and choices.
- **Syriatel's Initiative**: Syriatel aims to combat churn using predictive analytics.
- **Robust Churn Model**: Project goal is a predictive model for risk of churn.
- **Data-Driven Approach**: Utilizing historical customer data for effective categorization.
- **Proactive Solutions**: Model enables personalized retention actions and interventions.

### Data Understanding:

The dataset comprises vital customer interaction and behavior attributes in the telecom realm. It includes geographic location, account specifics, communication preferences, engagement levels, and call patterns. Data covers call durations, charges, service interactions, and the crucial "churn" indicator. This binary marker signifies service discontinuation. The dataset provides insights into behavior, usage patterns, and potential churn factors. This information forms the basis for predictive modeling to anticipate and mitigate customer churn in the telecom industry.

## Data Preparation

Identifying and handling missing values
Identifying and handling duplicates
Converting state column dummies to allow easy processing and dropping the area code column
Changing the international plan and voice mail plan to 1s and 0s
Dropped phone number column as it does not have any connection to churn column

## Data Visualization

### Visualizing count of churned customers

From the graph, roughly 500 customers have churned.

### Total Day Calls of Churned customers vs Retained Customers

Comparing call volumes of churned and retained customers reveals higher call activity among retained customers, possibly indicating stronger engagement and satisfaction with SyriaTel's services. This suggests a positive link between call activity and customer retention.

### International plans status for both churned and retained customers.

Comparing international plan presence among churned and retained customers, both show a low proportion of customers with international plans. This suggests international calling might not significantly impact customer churn for SyriaTel, as many customers may not use these services.

## Data Modeling

### Base Modeling

We first convert the target column to binary:
Splitting the data and target to test and train values.We then fit our logistic regression model on training set and evaluate its performance by calculating accuracy score.The model's F1 score of 0.36 indicates poor precision and recall, reflecting its inability to accurately detect positive cases. Class imbalance might be contributing to this performance. We then plot the ROC curve. 

##### Performing class balancing

After resampling, our new target has equal number of 0s and 1s. We did this by oversampling minority classes(churn) of our dataset.

### Random Forest Model

This is particularly suitable because it can provide insights into patterns, trends, and factors that contribute to the goals for this project. We got accuracy score of 0.94.
Here we perform hyperparameter tuning with GridSearchCV to improve the performance and generalization of your Random Forest model. Which allows us to find the best combination of hyperparameters that can lead to a more accurate and robust model.
We do feature importance analysis for understanding and improving our model. It  ultimately leads to a more effective and actionable model for your project's objectives.

### Decision Tree Classifier

1. **Classifier with default parameters**

Here we built a model with default parameters to assessed it's training and test accuracy.
From the results, the training accuracy is at 100% but the test accuracy is at 84%. This clearly shows that the model is overfitting. To solve this, I'll perform hyperparameter tuning using grid search and observe the possible changes.

2. **Tuned classifier and fitted with pipeline.**

Checking the Train and Test accuracy:
Training Accuracy = 89.3766461808604%
Testing Accuracy = 87.85607196401799%
From both accuracy scores, the training accuracy reduced from 100% to 88.35% while the test accuracy improved from  84% to 87%. The model is no longer overfitting after hyperparameter tuning with Grid Search.

These below are the **best parameters** for the scores provided:[{'tree__criterion': 'gini',
 'tree__max_depth': 10,
 'tree__min_samples_split': 5}]

### Bagging model

Use Bagging Classifier to build a model that accurately classify churn
The accuracy is 0.89 and the f1 score is 0.53
As seen with f1 score bagging model has greatly improved from Logistic Regression model but its not performing optimal so we can try improving it with the help GridSearchCV

### XGBoost model 

XGBoost to build a model that can accurately classify churn  on the features of the dataset!
Training Accuracy: 100.0%
Testing  accuracy: 95.2%
We then tune our XGBoost model using the grid search methodology.
Grid Search found the following optimal parameters: 
learning_rate: 1
max_depth: 5
n_estimators: 5

Training Accuracy: 96.92%
Testing accuracy: 94.9%
