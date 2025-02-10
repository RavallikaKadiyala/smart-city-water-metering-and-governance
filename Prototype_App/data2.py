import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load models
models = {
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "Linear Regression": "linear_regression_model.pkl",
    "Support Vector Machine": "svm_model.pkl"
}

def load_model(model_name):
    with open(models[model_name], 'rb') as file:
        return pickle.load(file)

# Load dataset
df = pd.read_csv("/Users/ravallikakadiyala/Downloads/capstone project/Updated_Water_Consumption_Dataset.csv")

# Extract required features dynamically from model
def get_required_features(model):
    try:
        return model.feature_names_in_.tolist()
    except AttributeError:
        return df.columns.tolist()  # Default to all columns if not available

# Encoding categorical variables
label_encoders = {}
categorical_cols = ["Borough", "Season", "Month"]
existing_cols = df.columns.tolist()
original_categorical_values = {}

for col in categorical_cols:
    if col in existing_cols:
        df[col] = df[col].astype(str).fillna("Unknown")  # Ensure no NaNs
        original_categorical_values[col] = df[col].unique().tolist()
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Ensure required features exist in dataset
def ensure_all_features(df, required_features):
    for feature in required_features:
        if feature not in df.columns:
            df[feature] = 0  # Default value for missing columns
    return df

# Sidebar UI
st.sidebar.title("Water Consumption Predictor")
st.sidebar.subheader("Team Members:")
st.sidebar.write("- Ravallika \n- Nawazuddin\n- Jay")

# Model selection
target_model = st.sidebar.selectbox("Select Model", list(models.keys()))
model = load_model(target_model)
required_features = get_required_features(model)

df = ensure_all_features(df, required_features)

# Main UI
st.title("Water Consumption Prediction")

# Row selection for autofill (Fix indexing issue)
row_number = st.number_input("Enter CSV Line Number to Auto-Fill", min_value=1, max_value=len(df), step=1, value=1)
data_examples = df.iloc[row_number - 1]  # Fixing the indexing issue

st.write("Enter the details below to predict water consumption (HCF):")

# Creating input fields in a 3x3 grid
user_inputs = {}
cols = st.columns(3)
i = 0

# Fields that should remain as text
text_fields = ["Development Name", "Account Name", "Location", "Meter AMR", "RC Code", "Funding Source",
               "AMP #", "Vendor Name", "Revenue Month", "Service Start Date", "Service End Date", 
               "Meter Number", "Estimated", "Rate Class", "Bill Analyzed", "Meter Scope"]  # Included Meter Scope

# Ensure autofill is 100% correct
for feature in required_features:
    if feature in existing_cols:
        if feature in categorical_cols:
            # Categorical dropdowns should match the correct value
            options = original_categorical_values.get(feature, ["Unknown"])
            current_value = str(data_examples[feature])
            selected_index = options.index(current_value) if current_value in options else 0
            user_inputs[feature] = cols[i % 3].selectbox(feature, options, index=selected_index)
        elif feature in text_fields:
            # Keep text-based fields unchanged
            user_inputs[feature] = cols[i % 3].text_input(feature, value=str(data_examples.get(feature, "")))
        else:
            # Ensure numerical fields are autofilled correctly
            user_inputs[feature] = cols[i % 3].number_input(feature, value=float(data_examples[feature]), format="%.4f")
    else:
        # Default to 0 if missing
        user_inputs[feature] = 0

    i += 1

# Convert categorical inputs using encoders before prediction
for feature in categorical_cols:
    if feature in user_inputs and feature in label_encoders:
        if user_inputs[feature] not in label_encoders[feature].classes_:
            label_encoders[feature].classes_ = np.append(label_encoders[feature].classes_, user_inputs[feature])
        user_inputs[feature] = label_encoders[feature].transform([user_inputs[feature]])[0]

# Convert numerical inputs to appropriate types
for feature in user_inputs:
    if feature not in text_fields:  # Only convert numeric fields
        try:
            user_inputs[feature] = float(user_inputs[feature])
        except ValueError:
            st.error(f"Invalid input for {feature}. Please enter a numerical value.")
            st.stop()

# Convert input data to DataFrame
input_df = pd.DataFrame([user_inputs])
input_df = ensure_all_features(input_df, required_features)
input_df = input_df[required_features]  # Ensure correct column order

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)
    
   # if "Consumption (HCF)" in df.columns:
        #valid_rows = df[required_features].notna().all(axis=1)
        #accuracy = model.score(df.loc[valid_rows, required_features], df.loc[valid_rows, "Consumption (HCF)"]) * 100  
        #st.info(f"Model Accuracy: {accuracy:.2f}%")
    
    st.success(f"Predicted Consumption (HCF): {prediction[0]:.2f}")
