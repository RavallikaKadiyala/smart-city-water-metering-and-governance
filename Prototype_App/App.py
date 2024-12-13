import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Function to preprocess the data
def preprocess_data(df):
    numeric_columns = ["# days", "Current Charges", "Consumption (HCF)", "Water&Sewer Charges"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df["Service Start Date"] = pd.to_datetime(df["Service Start Date"], errors='coerce')
    df["Service End Date"] = pd.to_datetime(df["Service End Date"], errors='coerce')
    
    df["Service Duration"] = (df["Service End Date"] - df["Service Start Date"]).dt.days
    df["Estimated"] = df["Estimated"].map({"Y": True, "N": False})
    
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    df["Borough"].fillna(df["Borough"].mode()[0], inplace=True)
    df["Rate Class"].fillna(df["Rate Class"].mode()[0], inplace=True)
    
    return df

# Function to prepare features and target
def prepare_features_and_target(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    categorical_cols = ["Borough", "Development Name", "Rate Class"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), ['# days', 'Consumption (HCF)', 'Water&Sewer Charges', 'Service Duration']),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, y

# Function to train models
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    st.write(f"{model_name} Performance Metrics:")
    st.write(f"Mean Squared Error: {mse:.4f}")
    st.write(f"Root Mean Squared Error: {rmse:.4f}")
    st.write(f"Mean Absolute Error: {mae:.4f}")
    st.write(f"R-squared Score: {r2:.4f}")
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

# Streamlit UI
def main():
    st.title('Water Consumption and Charges Analysis')

    st.sidebar.header('Upload CSV File')
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type='csv')

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.write("## Dataset Overview")
        st.write(df.head())

        # Preprocess the data
        df = preprocess_data(df)

        st.write("## Descriptive Statistics")
        st.write(df.describe())

        st.write("## Visualizations")

        # Consumption vs Charges by Borough
        plt.figure(figsize=(10, 8))
        sns.boxplot(x="Borough", y="Consumption (HCF)", data=df)
        plt.title("Water Consumption by Borough")
        plt.xticks(rotation=45)
        st.pyplot(plt)  # Correct method to display the plot

        # Estimated vs Non-Estimated Bills
        plt.figure(figsize=(10, 8))
        estimated_compare = df.groupby("Estimated")[["Current Charges", "Consumption (HCF)"]].mean()
        estimated_compare.plot(kind="bar", ax=plt.gca())
        plt.title("Average Charges and Consumption\nEstimated vs Non-Estimated")
        plt.xlabel("Estimated Bill")
        plt.ylabel("Average Value")
        plt.legend(loc='best')
        st.pyplot(plt)  # Correct method to display the plot

        # Consumption vs Charges Scatter
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x="Consumption (HCF)", y="Current Charges", hue="Borough", data=df, alpha=0.5)
        plt.title("Consumption vs Charges by Borough")
        st.pyplot(plt)  # Correct method to display the plot

        # Service Duration Impact
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x="Service Duration", y="Current Charges", hue="Borough", data=df, alpha=0.5)
        plt.title("Service Duration vs Charges")
        st.pyplot(plt)  # Correct method to display the plot

        # Model Selection
        st.sidebar.header("Model Selection")
        model_option = st.sidebar.selectbox('Choose a model', 
                                           ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting'])

        # Prepare data for modeling
        X, y = prepare_features_and_target(df, target_column='Current Charges')

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scaling
        scaler = StandardScaler(with_mean=False)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if model_option == 'Linear Regression':
            model = LinearRegression()
        elif model_option == 'Decision Tree':
            model = DecisionTreeRegressor(max_depth=10, random_state=42)
        elif model_option == 'Random Forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_option == 'Gradient Boosting':
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)

        # Train the selected model
        model = train_model(model, X_train_scaled, y_train)

        # Predictions
        y_pred = model.predict(X_test_scaled)

        # Evaluate model
        evaluate_model(y_test, y_pred, model_option)

if __name__ == "__main__":
    main()
