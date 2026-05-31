import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================
# Load Dataset
# ==========================

data = pd.read_csv("loan_data.csv")

# ==========================
# Handle Missing Values
# ==========================

data['Gender'] = data['Gender'].fillna(
    data['Gender'].mode()[0]
)

data['Married'] = data['Married'].fillna(
    data['Married'].mode()[0]
)

data['Dependents'] = data['Dependents'].fillna(
    data['Dependents'].mode()[0]
)

data['Self_Employed'] = data['Self_Employed'].fillna(
    data['Self_Employed'].mode()[0]
)

data['LoanAmount'] = data['LoanAmount'].fillna(
    data['LoanAmount'].median()
)

data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(
    data['Loan_Amount_Term'].median()
)

data['Credit_History'] = data['Credit_History'].fillna(
    data['Credit_History'].mode()[0]
)

# ==========================
# Convert Categorical Data
# ==========================

data['Gender'] = data['Gender'].map({
    'Male': 1,
    'Female': 0
})

data['Married'] = data['Married'].map({
    'Yes': 1,
    'No': 0
})

data['Education'] = data['Education'].map({
    'Graduate': 1,
    'Not Graduate': 0
})

data['Self_Employed'] = data['Self_Employed'].map({
    'Yes': 1,
    'No': 0
})

# Convert Dependents

data['Dependents'] = (
    data['Dependents']
    .astype(str)
    .str.replace('+', '', regex=False)
)

data['Dependents'] = pd.to_numeric(
    data['Dependents'],
    errors='coerce'
)

data['Dependents'].fillna(
    data['Dependents'].median(),
    inplace=True
)

data['Property_Area'] = data['Property_Area'].map({
    'Rural': 0,
    'Semiurban': 1,
    'Urban': 2
})

data['Loan_Status'] = data['Loan_Status'].map({
    'Y': 1,
    'N': 0
})

# ==========================
# Features & Target
# ==========================

X = data.drop('Loan_Status', axis=1)

y = data['Loan_Status']

# ==========================
# Train-Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================
# Train Model
# ==========================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
print("\nData Types:\n")
print(X.dtypes)
model.fit(X_train, y_train)

# ==========================
# Accuracy
# ==========================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nModel Accuracy:",
      round(accuracy * 100, 2), "%")

# ==========================
# Feature Importance
# ==========================

print("\nFeature Importance:\n")

for feature, importance in zip(
        X.columns,
        model.feature_importances_):

    print(
        feature,
        ":",
        round(importance, 4)
    )

# ==========================
# Save Model
# ==========================

pickle.dump(
    model,
    open("model.pkl", "wb")
)

print("\nmodel.pkl created successfully")