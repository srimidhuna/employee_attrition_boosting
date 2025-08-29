import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    return df

df = load_data()
df = df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"])

important_features = [
    "Age", "MonthlyIncome", "JobRole", "OverTime", "DistanceFromHome"
]

X = df[important_features]
y = df["Attrition"]

X = X.copy()
encoders = {}
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

le_target = LabelEncoder()
y = le_target.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

st.title("HR Employee Attrition Prediction")
st.write("Enter employee details below to predict attrition:")

def user_input_features():
    data = {}
    for col in important_features:
        if df[col].dtype == "object":
            data[col] = st.selectbox(col, df[col].unique())
        else:
            data[col] = st.number_input(
                col,
                min_value=float(df[col].min()),
                max_value=float(df[col].max()),
                value=float(df[col].mean())
            )
    return pd.DataFrame([data])

input_df = user_input_features()

if st.button("Predict Attrition"):
    encoded_input = input_df.copy()
    for col in encoded_input.select_dtypes(include="object").columns:
        le = encoders[col]
        encoded_input[col] = le.transform(encoded_input[col])
    
    prediction = model.predict(encoded_input)

    st.subheader("Prediction")
    st.success("Yes - Employee will leave" if prediction[0] == 1 else "No - Employee will stay")

