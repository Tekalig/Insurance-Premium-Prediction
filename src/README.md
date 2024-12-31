# Source Code (`src`)

This folder contains the source code for the project. The code is modularized into various components to ensure scalability, maintainability, and reusability. Each file is designed to perform specific tasks related to data processing, visualization, modeling, and evaluation.

## Folder Structure

```
src/
├── data_processing.py        # Handles data loading, cleaning, and preparation.
├── feature_engineering.py    # Performs feature creation and transformation.
├── data_visualization.py     # Contains functions for visualizing data and insights.
├── model_building.py         # Implements machine learning models.
├── model_evaluation.py       # Evaluates model performance using metrics.
├── interpretability.py       # Provides tools for model interpretation (e.g., SHAP, LIME).
└── utils.py                  # Contains utility functions shared across modules.
```

## Modules

### 1. `data_processing.py`
This module handles:
- Loading data from various sources (e.g., CSV, databases).
- Cleaning data (e.g., handling missing values, duplicates).
- Encoding categorical variables and normalizing numerical features.
- Splitting data into training and testing sets.

### 2. `feature_engineering.py`
This module focuses on:
- Creating new features that enhance predictive power.
- Transforming existing features (e.g., scaling, log transformations).
- Selecting the most relevant features for modeling.

### 3. `data_visualization.py`
This module provides:
- Functions for exploratory data analysis (EDA).
- Visualization tools for insights (e.g., bar charts, scatter plots, heatmaps).
- Plots to compare model performances.

### 4. `model_building.py`
This module implements:
- Statistical and machine learning models (e.g., Linear Regression, Random Forests, XGBoost).
- Parameter tuning for optimal model performance.
- Pipelines for automating model training.

### 5. `model_evaluation.py`
This module evaluates model performance using:
- Metrics like accuracy, precision, recall, F1-score, and RMSE.
- Confusion matrices and ROC curves for classification tasks.
- Cross-validation to ensure model robustness.

### 6. `interpretability.py`
This module focuses on:
- Explaining model predictions using SHAP and LIME.
- Identifying key features that influence predictions.
- Visualizing feature importance for better insights.

### 7. `utils.py`
This module contains reusable utility functions such as:
- File and path management.
- Data sampling and preprocessing helpers.
- Logging and error handling.

## Usage
To use a specific module, import it in your script or notebook:

```python
from src.data_processing import load_data, clean_data
from src.model_building import train_model
```

## Contribution Guidelines
- Follow the modular structure and avoid adding redundant code.
- Document functions clearly with docstrings.
- Test new code thoroughly before committing changes.

## Contact
For questions or issues, please contact **Tekalign Mesfin**.
