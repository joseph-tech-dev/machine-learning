import joblib
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Reading the csv
telco = pd.read_csv("Telco-customer-churn.csv")
# Replacing yes:1 and no:0
telco.replace({"Yes": 1, "No": 0, "Male":1, "Female":0, "No phone service":0, "No internet service":0}, inplace=True)
# One-hot encoding multi-categorical columns
telco = pd.get_dummies(telco, columns=["Contract", "PaymentMethod","InternetService"], drop_first=True)


# Convert TotalCharges (could have whitespace or be read as str)
telco["TotalCharges"] = pd.to_numeric(telco["TotalCharges"], errors='coerce')
# Dropping NaN
telco.dropna(inplace=True)
# Feature Selection
X = telco.drop(columns=["customerID", "Churn"])
y = telco["Churn"]

# Splitting Dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Fitting the data  for training and evaluation
model = LogisticRegression(
    max_iter=4000,
    n_jobs=-1)
model.fit(x_train, y_train)
joblib.dump(model, "Telco_custom_churn_model.joblib")
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred).__round__(2)*100,"%")
print("Precision:", precision_score(y_test, y_pred).__round__(2)*100,"%")
print("Recall:",recall_score(y_test, y_pred).__round__(2)*100,"%")
print("F1 Score:",f1_score(y_test, y_pred).__round__(2)*100,"%")
print("===================================")

