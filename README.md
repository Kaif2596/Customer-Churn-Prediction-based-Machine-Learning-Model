
# Customer Churn Prediction based Machine Learning Model



## 📋  Description

This repository contains a Machine Learning model designed to predict Customer Churn for subscription-based services. By analyzing historical customer data, the model identifies users at high risk of canceling their subscriptions, enabling proactive intervention to retain them.  This model and repository provides an end-to-end solution including data preprocessing, model building and evaluation.

## 🚀 Key Features

*   **Churn Prediction:** Predicts the likelihood of a customer churning (canceling their subscription).
*   **Data Analysis:** Analyzes historical customer data to identify patterns and factors contributing to churn.
*   **Data Preprocessing:** Handles missing values using appropriate imputation techniques, ensuring data quality for model training.
*   **Model Explainability:** Provides insights into the most significant factors driving customer churn.
*   **Ease of Use:**  Well-documented code and clear instructions for setup and execution.
*   **Flexibility:** The model can be adapted and retrained with new data as needed.
*   **Modular Design:**  The code is designed in a modular fashion for easy understanding and extension.
## 🛠️ Installation and Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Kaif2596/Customer-Churn-Prediction-based-Machine-Learning-Model.git
    cd customer-churn-prediction
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    venv\Scripts\activate.bat # On Windows
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

    (Create a `requirements.txt` file with the following content, adjusting versions as needed.)

    ```
    pandas==1.5.3
    scikit-learn==1.2.2
    matplotlib==3.7.0
    seaborn==0.12.2
    numpy==1.23.5
    # Add any other dependencies your code uses
## 🧠 How It Works

### Data Preparation:
1.  **Dataset:**  The model is designed to work with customer data in a CSV format. My dataset (`Churn_Modelling.csv`) is included in the repository. 


2.  **Data Location:** Place your dataset in the `"D:\Credit Card Transaction Fraud Detection\Churn_Modelling.csv"` directory. 

    ```bash
    mkdir data
    # "D:\Credit Card Transaction Fraud Detection\Churn_Modelling.csv"

### Running the Model

1.  **Execute the main script:**

    ```bash
    Bank Customer Churn Prediction.ipynb
    ```

 

2.  **Output:**  The script will perform the following:

    *   Loads and preprocesses the data.
    *   Trains the machine learning model.
    *   Evaluates the model's performance.
    *   Provides insights into the key factors contributing to churn.
    *   Saves the trained model to a file (`churn_model.pkl`).

## Data Dictionary

The dataset should contain the following columns (adjust names if necessary to match your data):

*   **RowNumber:** Unique identifier for each row (customer).
*   **CustomerId:** Unique identifier for each customer.
*   **Surname:** Customer's last name.
*   **CreditScore:** Credit score of the customer.
*   **Geography:** Country where the customer resides (France, Spain, Germany).
*   **Gender:** Customer's gender (Male, Female).
*   **Age:** Age of the customer.
*   **Tenure:** Number of years the customer has been with the service.
*   **Balance:** Account balance of the customer.
*   **NumOfProducts:** Number of products the customer uses.
*   **HasCrCard:** Whether the customer has a credit card (1 = yes, 0 = no).
*   **IsActiveMember:** Whether the customer is an active member (1 = yes, 0 = no).
*   **EstimatedSalary:** Estimated salary of the customer.
*   **Exited (Target Variable):** Whether the customer churned (1 = yes, 0 = no).

## Model Details

### Model Selection

*   The current implementation uses a Random Forest and Logistic Regression classifier for churn prediction.


   

### Model Training and Evaluation

*   The model is trained on a [describe train/test split, e.g., 80/20] split of the historical customer data.
*   [Describe any cross-validation strategy used, e.g., 5-fold cross-validation].

### Metrics

The model's performance is evaluated using the following metrics:

*   **Accuracy:** Overall correctness of the model's predictions.
*   **Precision:**  Of all the customers predicted to churn, what proportion actually churned?
*   **Recall:** Of all the customers that actually churned, what proportion did the model correctly identify?
*   **F1-score:** The harmonic mean of precision and recall, providing a balanced measure of the model's performance.
*   **AUC-ROC:** Area Under the Receiver Operating Characteristic curve, which measures the model's ability to distinguish between churners and non-churners.



## 📊 Demo

![image alt](https://github.com/Kaif2596/Customer-Churn-Prediction-based-Machine-Learning-Model/blob/main/Screenshot%20(20).png)


## 📈 Future Enhancement:

*   **Hyperparameter Tuning:**  Explore different hyperparameter optimization techniques to improve the model's performance (e.g., GridSearchCV, RandomizedSearchCV).
*   **Advanced Feature Engineering:** Experiment with more sophisticated feature engineering approaches, such as creating interaction terms or using domain-specific knowledge.
*   **Explore Other Algorithms:** Evaluate the performance of other machine learning algorithms, such as XGBoost, LightGBM, or neural networks.
*   **Deployment:**  Containerize and deploy the model as a web service for real-time prediction.
*   **A/B Testing:** Test the model in a live environment with A/B testing.
