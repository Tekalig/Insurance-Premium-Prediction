import pandas as pd

def load_data(file_path):
    """Load the dataset from a text file with '|' as delimiter."""
    df = pd.read_csv(file_path, delimiter="|",
                     low_memory=False)  # Set low_memory=False to ensure that datatype inference is done correctly
    return df


def clean_data(df):
    """
    Clean the data by handling missing values and formatting issues.
    - Drop columns with more than 50% missing values.
    - Drop rows with missing values in remaining columns.
    """
    # Define the threshold for dropping columns
    threshold = 0.5  # 50%

    # Calculate the percentage of missing values for each column
    missing_percentages = df.isnull().mean()

    # Drop columns with more than the threshold of missing values
    columns_to_drop = missing_percentages[missing_percentages > threshold].index
    df = df.drop(columns=columns_to_drop)

    # Drop rows with missing values in the remaining columns
    df = df.dropna()

    return df


def preprocess_data(df):
    """Preprocess data: Convert data types, handle categorical variables, etc."""
    # Convert relevant columns to datetime or categorical
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
    df['Citizenship'] = df['Citizenship'].fillna('Unknown')  # Handle missing Citizenship data
    # Additional preprocessing can be done here
    return df


def check_missing_values(df):
    """Check for missing values in the dataset and provide a summary."""

    # Check for missing values
    missing_data = df.isnull().sum()

    # Get percentage of missing data for each column
    missing_percentage = (missing_data / len(df)) * 100

    # Create a summary DataFrame with missing count and percentage
    missing_summary = pd.DataFrame({
        'Missing Values': missing_data,
        'Percentage': missing_percentage
    })

    # Filter out columns that have no missing values
    missing_summary = missing_summary[missing_summary['Missing Values'] > 0]

    print("\nMissing Values Summary:")
    print(missing_summary)