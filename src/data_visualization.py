import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(df, column):
    """Plot histogram for a numerical column."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

def plot_boxplot(df, column):
    """Plot boxplot to detect outliers."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot of {column}")
    plt.show()

def plot_scatter(df, x_column, y_column):
    """Plot scatter plot for two numerical columns."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[x_column], y=df[y_column])
    plt.title(f"Scatter plot between {x_column} and {y_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()


def univariate_analysis(df):
    """Perform uni-variate analysis: Plot histograms for numerical columns and bar charts for categorical columns."""

    # Uni variate analysis for numerical columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Plot histograms for numerical columns
    for column in numerical_columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=True, bins=30, color='skyblue')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    # Uni-variate analysis for categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    # Plot bar charts for categorical columns
    for column in categorical_columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=column, palette='Set2')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()


def bivariate_analysis(df):
    """Perform bivariate analysis: Explore relationships between TotalPremium, TotalClaims, and ZipCode."""

    # Step 1: Correlation Matrix for numerical variables
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Calculate the correlation matrix
    correlation_matrix = df[numerical_columns].corr()

    # Step 2: Create subplots for Correlation Matrix and Scatter Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot the Correlation Matrix on the first subplot (axes[0])
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=axes[0])
    axes[0].set_title('Correlation Matrix')

    # Plot the Scatter Plot for TotalPremium vs TotalClaims on the second subplot (axes[1])
    sns.scatterplot(data=df, x='TotalPremium', y='TotalClaims', hue='PostalCode', palette='viridis', ax=axes[1])
    axes[1].set_title('TotalPremium vs TotalClaims by ZipCode')
    axes[1].set_xlabel('TotalPremium')
    axes[1].set_ylabel('TotalClaims')

    # Display the plots
    plt.tight_layout()
    plt.show()


def data_comparison(df):
    """Perform data comparison: Compare trends over geography (Province or PostalCode)."""

    # Step 1: Group data by 'Province' and calculate the mean for relevant columns
    # This can be adjusted based on the available geographic columns in the dataset
    geographic_columns = ['Province', 'CoverType', 'make', 'TotalPremium']

    # Aggregate data by 'Province' and 'CoverType'
    province_cover_type = df.groupby(['Province', 'CoverType'], observed=True).agg(
        {'TotalPremium': 'mean', 'make': 'nunique'}).reset_index()

    # Step 2: Plot the trends for TotalPremium by Province and CoverType
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Province', y='TotalPremium', hue='CoverType', data=province_cover_type)
    plt.title('Average Total Premium by Province and Cover Type')
    plt.xlabel('Province')
    plt.ylabel('Average Total Premium')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Step 3: Compare the number of unique auto makes across provinces
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Province', y='make', data=province_cover_type, hue='CoverType')
    plt.title('Number of Unique Auto Makes by Province and Cover Type')
    plt.xlabel('Province')
    plt.ylabel('Number of Unique Auto Makes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def outlier_detection(df, numerical_columns):
    """Detect outliers using box plots for numerical columns."""

    # Step 1: Plot box plots for the numerical columns
    num_columns = len(numerical_columns)
    rows = 3  # Set the number of rows to 3
    cols = (num_columns // rows) + (num_columns % rows > 0)  # Calculate the number of columns needed

    plt.figure(figsize=(15, rows * 5))  # Adjust figure size based on rows

    for i, col in enumerate(numerical_columns, 1):
        plt.subplot(rows, cols, i)  # Create subplots with specified rows and columns
        sns.boxplot(x=df[col])
        plt.title(f'Box Plot for {col}')
        plt.xlabel(col)

    plt.tight_layout()
    plt.show()



def create_visualizations(df):
    """Create 3 creative and beautiful plots for EDA insights."""

    # 1. Correlation Heatmap
    plt.figure(figsize=(10, 6))
    corr = df.corr(numeric_only=True)  # Compute correlation matrix
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar=True)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    plt.show()

    # 2. Distribution of Premiums and Claims by Vehicle Type (Box Plot)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)  # Subplot for TotalPremium
    sns.boxplot(x='VehicleType', y='TotalPremium', data=df)
    plt.title('Distribution of TotalPremium by Vehicle Type')
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)  # Subplot for TotalClaims
    sns.boxplot(x='VehicleType', y='TotalClaims', data=df)
    plt.title('Distribution of TotalClaims by Vehicle Type')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # 3. Trends Over Geography (Average Premiums per Province)
    plt.figure(figsize=(12, 6))
    avg_premium_by_province = df.groupby('Province', observed=True)['TotalPremium'].mean().sort_values(ascending=False)
    avg_premium_by_province.plot(kind='bar', color='skyblue')
    plt.title('Average TotalPremium by Province')
    plt.xlabel('Province')
    plt.ylabel('Average TotalPremium')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
