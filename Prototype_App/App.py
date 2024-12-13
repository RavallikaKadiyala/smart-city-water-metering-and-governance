import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, accuracy_score,
    precision_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit Page Configuration
st.set_page_config(page_title="Water Consumption App", layout="wide")

# Title of the App
st.title("Water Consumption and Cost Analysis")

# Upload Dataset
uploaded_file = st.file_uploader("Upload Water Consumption Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)

    # Data Overview
    if st.checkbox("Show Dataset Overview"):
        st.write("Dataset Overview:")
        st.write(data.head())
        st.write(data.info())

    # Handle Missing Values
    data = data.dropna()

    # Convert Dates to Datetime
    data['Service Start Date'] = pd.to_datetime(data['Service Start Date'])
    data['Service End Date'] = pd.to_datetime(data['Service End Date'])

    # Feature Engineering: Consumption per Day
    data['Consumption_per_day'] = data['Consumption (HCF)'] / data['# days']

    # Remove Infinite and Large Values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.dropna()

    # Select Relevant Columns
    features = ['Consumption (HCF)', 'Consumption_per_day', 'Current Charges']
    target = 'Estimated'  # Classification Target (Y/N)

    # Encode Target Variable
    data['Estimated'] = data['Estimated'].apply(lambda x: 1 if x == 'Y' else 0)

    # Split Dataset into Features and Target
    X = data[features]
    y = data[target]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Model: Linear Regression ---
    regressor = LinearRegression()
    regressor.fit(X_train[['Consumption (HCF)']], X_train['Current Charges'])
    y_pred_lr = regressor.predict(X_test[['Consumption (HCF)']])
    mae = mean_absolute_error(X_test['Current Charges'], y_pred_lr)
    mse = mean_squared_error(X_test['Current Charges'], y_pred_lr)
    rmse = np.sqrt(mse)

    st.write(f"Linear Regression - MAE: {mae}, MSE: {mse}, RMSE: {rmse}")

    # --- Model: Decision Tree Classifier ---
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred_tree = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_tree)
    precision = precision_score(y_test, y_pred_tree)
    roc_auc = roc_auc_score(y_test, y_pred_tree)

    st.write(f"Decision Tree - Accuracy: {accuracy}, Precision: {precision}, AUC: {roc_auc}")

    # --- K-Means Clustering ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    data['Cluster'] = clusters

    # Visualize Clusters
    if st.checkbox("Show K-Means Clustering Visualization"):
        st.write("K-Means Clustering Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=data, x='Consumption (HCF)', y='Current Charges', hue='Cluster', palette='viridis', ax=ax)
        st.pyplot(fig)

    # --- Time Series Analysis ---
    data['Revenue Month'] = pd.to_datetime(data['Revenue Month'])
    monthly_data = data.groupby('Revenue Month')['Consumption (HCF)'].sum().reset_index()

    # Time Series Decomposition
    if st.checkbox("Show Time Series Decomposition"):
        decomposition = sm.tsa.seasonal_decompose(monthly_data['Consumption (HCF)'], period=12)
        st.write("Time Series Decomposition")
        decomposition.plot()
        st.pyplot(plt)

    # ARIMA Model
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(monthly_data['Consumption (HCF)'], order=(1, 1, 1))
    results = model.fit()

    st.write("ARIMA Model Summary:")
    st.write(results.summary())

    # --- Correlation Heatmap ---
    numeric_data = data.select_dtypes(include=[np.number])

    if st.checkbox("Show Correlation Heatmap"):
        st.write("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # --- Regression Plot ---
    if st.checkbox("Show Linear Regression Plot"):
        st.write("Linear Regression: Charges vs Consumption")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(x='Consumption (HCF)', y='Current Charges', data=data, line_kws={"color": "red"}, ax=ax)
        st.pyplot(fig)

    # --- Display Evaluation Summary ---
    st.write(f"Linear Regression - MAE: {mae}, MSE: {mse}, RMSE: {rmse}")
    st.write(f"Decision Tree - Accuracy: {accuracy}, Precision: {precision}, AUC: {roc_auc}")

    st.write("Decision Tree Classification Report:")
    from sklearn.metrics import classification_report
    st.text(classification_report(y_test, y_pred_tree))
