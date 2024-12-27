import pandas as pd


def calculate_descriptive_statistics(df, numeric_columns):
    """Calculate descriptive statistics and variability for numerical features."""

    # Descriptive statistics for numerical features
    descriptive_stats = df[numeric_columns].describe()
    print("Descriptive Statistics for Numerical Features:")
    print(descriptive_stats)

    # Calculate the variability (standard deviation) for key numerical features
    variability = df[numeric_columns].std()
    print("\nVariability (Standard Deviation) for Numerical Features:")
    print(variability)

    return descriptive_stats, variability


def review_data_structure(df):
    """Review the data structure and confirm if categorical, dates, etc. are properly formatted."""

    # Check the data types of each column
    data_types = df.dtypes
    print("Data Types of Each Column:")
    print(data_types)

    # Identify columns that should be categorical
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    print("\nCategorical Columns:")
    print(categorical_columns)

    # Identify columns that should be dates
    date_columns = df.select_dtypes(include=['datetime']).columns
    print("\nDate Columns:")
    print(date_columns)

    # Identify numerical columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    print("\nNumerical Columns:")
    print(numeric_columns)

    # Check if any categorical columns should be converted to category dtype
    for col in categorical_columns:
        if df[col].dtype != 'category':
            print(f"\nConverting {col} to 'category' dtype")
            df[col] = df[col].astype('category')

    # Check if any date columns need conversion to datetime dtype
    for col in date_columns:
        if df[col].dtype != 'datetime':
            print(f"\nConverting {col} to 'datetime' dtype")
            df[col] = pd.to_datetime(df[col])

    return df