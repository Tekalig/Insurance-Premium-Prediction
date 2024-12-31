import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import shap

def prepare_data(data):
    """
    Prepare the data by handling missing values, encoding categorical variables,
    and splitting into features and target variable.

    Parameters:
    data (DataFrame): The dataset containing insurance policy data.

    Returns:
    tuple: X_train, X_test, y_train, y_test
    """
    # Handle missing values
    data = data.dropna()

    # Feature engineering
    data['VehicleAge'] = 2024 - data['RegistrationYear']  # Assuming current year is 2024
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).drop(columns=['TotalPremium'])
    y = data['TotalPremium']
    # Encoding categorical variables
    categorical_columns = ['Province', 'IsVATRegistered', 'VehicleType', 'CoverCategory']
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_features = encoder.fit_transform(data[categorical_columns])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

    # Combine the encoded features with the rest of the dataset
    data = pd.concat([numeric_columns, encoded_df], axis=1)

    # Defining features (X) and target (y)
    X = data.drop(columns=['TotalClaims'])  # Adjust target variable as needed

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def build_models(X_train, X_test, y_train, y_test):
    """
    Build and evaluate different models: Linear Regression, Decision Trees, Random Forest, XGBoost.

    Parameters:
    X_train, X_test, y_train, y_test (DataFrames): Train and test datasets.

    Returns:
    dict: Dictionary containing model performance metrics.
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'XGBoost': GradientBoostingRegressor()
    }

    model_results = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        model_results[model_name] = {'MSE': mse, 'MAE': mae, 'R2': r2}

    return model_results, models


def plot_feature_importance(model, X_train):
    """
    Plot feature importance using SHAP (SHapley Additive exPlanations).

    Parameters:
    model (Model): The trained model (e.g., RandomForest, XGBoost).
    X_train (DataFrame): Training dataset features.
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    # Summary plot of feature importance
    shap.summary_plot(shap_values, X_train)


def evaluate_models(model_results):
    """
    Print and compare the evaluation metrics for all models.

    Parameters:
    model_results (dict): Dictionary containing model performance metrics.
    """
    for model_name, metrics in model_results.items():
        print(f"Model: {model_name}")
        print(f"Mean Squared Error (MSE): {metrics['MSE']}")
        print(f"Mean Absolute Error (MAE): {metrics['MAE']}")
        print(f"R-Squared (R2): {metrics['R2']}")
        print('-' * 50)
