import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# 1. Load the data
df = pd.read_csv("D:\Credit Card Transaction Fraud Detection\Churn_Modelling.csv") # Replace with your actual file path
print(df.head())

# 2. Data Exploration and Preprocessing

# Drop unnecessary columns like RowNumber, CustomerId, and Surname
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
print("\nData Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())  #Shows number of missing values in each column

# Imputation strategy
#Numerical features: mean imputation
#Categorical features: mode imputation
numerical_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(exclude=np.number).columns

numerical_imputer = SimpleImputer(strategy='mean') #Mean is common
df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])

categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

# Verify that missing values have been handled
print("\nMissing Values After Imputation:")
print(df.isnull().sum())

# 3. Feature Engineering and Encoding

# Create a ColumnTransformer to apply different transformations to different columns
numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary'] #List all numerical columns
categorical_features = ['Geography', 'Gender'] #Categorical columns

# Using make_column_transformer() for conciseness
ct = ColumnTransformer(
    [
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # 'handle_unknown' important to avoid errors
    ],
    remainder='passthrough' # Include the remaining columns
)

# 4. Model Building

# Define features (X) and target (y)
X = df.drop('Exited', axis=1)
y = df['Exited']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% training, 20% testing

# Logistic Regression
pipeline_lr = Pipeline([
    ('transformer', ct),
    ('classifier', LogisticRegression(random_state=42))
])
pipeline_rf = Pipeline([
    ('transformer', ct),
    ('classifier', RandomForestClassifier(random_state=42))
])
#Hyperparameter Tuning using GridSearchCV
#Define the grid of hyperparameters to search
# Logistic Regression parameters
param_grid_lr = {
    'classifier__penalty': ['l1', 'l2'],
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__solver': ['liblinear', 'saga']
}

# Random Forest parameters
param_grid_rf = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [4, 6, 8, 10],
    'classifier__min_samples_split': [2, 4, 6]
}
grid_search_lr = GridSearchCV(pipeline_lr, param_grid_lr, scoring='roc_auc', cv=3)
grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, scoring='roc_auc', cv=3) #CV=3 is a common value

# Fit the GridSearchCV objects
grid_search_lr.fit(X_train, y_train)
grid_search_rf.fit(X_train, y_train)

# Print the best parameters
print("Best parameters for Logistic Regression:", grid_search_lr.best_params_)
print("Best parameters for Random Forest:", grid_search_rf.best_params_)

# Make predictions on the test data
y_pred_lr = grid_search_lr.predict(X_test)
y_pred_rf = grid_search_rf.predict(X_test)

# Calculate evaluation metrics
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
roc_auc_lr = roc_auc_score(y_test, grid_search_lr.predict_proba(X_test)[:, 1])

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, grid_search_rf.predict_proba(X_test)[:, 1])

# Print the evaluation metrics
print("Logistic Regression Metrics:")
print("Accuracy:", accuracy_lr)
print("Precision:", precision_lr)
print("Recall:", recall_lr)
print("F1 Score:", f1_lr)
print("AUC-ROC:", roc_auc_lr)

print("\nRandom Forest Metrics:")
print("Accuracy:", accuracy_rf)
print("Precision:", precision_rf)
print("Recall:", recall_rf)
print("F1 Score:", f1_rf)
print("AUC-ROC:", roc_auc_rf)

#Feature importance:
rf_model = grid_search_rf.best_estimator_.named_steps['classifier']
feature_importances = rf_model.feature_importances_

# Ensure the ColumnTransformer is fitted
ct.fit(X_train)  # Replace X_train with your actual dataset

# Now access named transformers
feature_names = (
    ct.named_transformers_['num'].get_feature_names_out(numerical_features).tolist() +
    ct.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist() +
    ['HasCrCard', 'IsActiveMember', 'Tenure', 'Balance', 'NumOfProducts', 'CreditScore']
)

feature_importance_dict = dict(zip(feature_names, feature_importances))

# Sort the features by importance
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

#Print top 10 features
print('\nTop 10 features:')
for feature, importance in sorted_features[:10]:
    print(f"{feature}: {importance}")

# Visualization of feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=[importance for feature, importance in sorted_features[:10]], y=[feature for feature, importance in sorted_features[:10]])
plt.title("Top 10 Feature Importances")
plt.show()