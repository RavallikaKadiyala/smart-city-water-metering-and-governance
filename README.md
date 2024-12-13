# smart-city-water-metering-and-governance
This project focuses on developing a smart water metering system for urban areas, aiming to improve water usage efficiency and governance. By analyzing historical water consumption data, the project uses machine learning models to predict consumption patterns, optimize billing accuracy, and enhance resource management. The goal is to provide insights for better water distribution, reduce wastage, and support sustainable water practices in smart cities. The project also includes data preprocessing, exploratory data analysis (EDA), and model comparisons for improved decision-making in water metering and billing systems.

## Features

- **Data Upload**: Upload a CSV file containing water consumption data.
- **Data Preprocessing**: The app handles missing values, converts columns to appropriate data types, and creates new features like service duration.
- **Data Visualizations**: Interactive charts for visualizing water consumption trends, charges, and comparisons.
- **Model Training and Evaluation**: Allows users to select different machine learning models (Linear Regression, Decision Tree, Random Forest, Gradient Boosting) to predict water charges. The models are evaluated based on performance metrics like MSE, RMSE, MAE, and R².

## Requirements

- Python 3.x
- Streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Installation

1. Clone or download the repository to your local machine.
2. Install the required dependencies using the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

### Running on Another Device

To run the app on another device:

1. **Clone the repository** on the new device or transfer the project folder.
2. **Install Python** if it’s not installed already. Download it from [python.org](https://www.python.org/downloads/).
3. **Install dependencies** by navigating to the project directory and running:

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the app** by executing the following command from the project directory:

    ```bash
    streamlit run path/to/your/project/Prototype_App/App.py
    ```

   Replace `path/to/your/project` with the actual file path on the new device.

5. The app will automatically open in your default web browser, or you can manually visit the provided localhost URL (usually `http://localhost:8501`).

## Data Format

Ensure your CSV file follows this format:

- `Service Start Date`: Date when the water service started.
- `Service End Date`: Date when the water service ended.
- `Borough`: The borough of the customer.
- `Rate Class`: The rate class of the customer.
- `Current Charges`: The water charges.
- `Consumption (HCF)`: The water consumption in Hundred Cubic Feet (HCF).
- `Estimated`: A boolean flag indicating if the charges are estimated ("Y" for Yes, "N" for No).
- `# days`: Duration in days for the service.