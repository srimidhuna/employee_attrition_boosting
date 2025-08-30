Employee Attrition Prediction using XGBoost
📌 Project Overview

Employee attrition (voluntary or involuntary exits) is a major challenge for organizations. Predicting attrition can help HR teams take proactive measures to retain talent.
This project applies XGBoost, a gradient boosting algorithm, to classify whether an employee will leave the organization based on features like work environment, satisfaction, and salary.

Dataset:  https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset 

📊 Dataset Description

The dataset contains employee details such as:

Demographics: Age, Gender, Education, Marital Status

Job-related: Department, Job Role, Job Level, Years at Company

Work Environment: Work-Life Balance, Job Involvement, Overtime

Compensation: Monthly Income, Salary Hike, Stock Option Level

Target Variable: Attrition (Yes = employee left, No = employee stayed)

⚙️ Methodology
1. Data Preprocessing

Load dataset from CSV

Handle categorical variables using Label Encoding / One-Hot Encoding

Normalize/scale numerical features where necessary

Split dataset into train and test sets

2. Model Training (XGBoost)

Use XGBoost Classifier from xgboost library

Perform 5-fold cross-validation

Hyperparameter tuning for:

learning_rate

n_estimators

max_depth

3. Model Evaluation

Evaluate model performance using:

Accuracy

Precision

Recall

F1-score

ROC-AUC

📦 Requirements

Install required Python packages:

pip install pandas numpy scikit-learn xgboost matplotlib seaborn

🚀 How to Run

Clone this repository or download the project files

Place the dataset (WA_Fn-UseC_-HR-Employee-Attrition.csv) in the project folder

Run the script:

python attrition_xgboost.py


The script will:

Train an XGBoost classifier with 5-fold cross-validation

Print classification metrics

Plot the ROC curve

📈 Sample Output

Accuracy: ~85%

Precision: 0.79

Recall: 0.72

F1-Score: 0.75

ROC-AUC: 0.89

(Values may vary depending on hyperparameter tuning)

📌 Future Improvements

Perform feature importance analysis

Handle class imbalance using SMOTE or class weighting

Deploy as an HR dashboard (Flask/Streamlit)

Compare with other algorithms (Random Forest, Logistic Regression, etc.)


APP LINK: https://employeeattritionboosting-h9n759rfp7sqpn7eaanaem.streamlit.app/
