import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("loan_data.csv")

# Fill missing values

data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
data['Married'].fillna(data['Married'].mode()[0], inplace=True)
data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].median(), inplace=True)
data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)

# Remove Loan_ID

data.drop('Loan_ID', axis=1, inplace=True)

# Encode categorical values

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

data['Property_Area'] = data['Property_Area'].map({
    'Rural': 0,
    'Semiurban': 1,
    'Urban': 2
})

data['Dependents'] = data['Dependents'].replace({
    '0': 0,
    '1': 1,
    '2': 2,
    '3+': 3
})

data['Loan_Status'] = data['Loan_Status'].map({
    'Y': 1,
    'N': 0
})

# Features and Target

X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)
print(X.columns)
model.fit(X_train, y_train)

# Accuracy

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("\nAccuracy :", round(accuracy * 100, 2), "%")

print("\nFeature Importance:\n")

for feature, importance in zip(
        X.columns,
        model.feature_importances_):
    print(
        feature,
        ":",
        round(importance, 4)
    )

# Save model

pickle.dump(model, open('model.pkl', 'wb'))

print("\nmodel.pkl created successfully")