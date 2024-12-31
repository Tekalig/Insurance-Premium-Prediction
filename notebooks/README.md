# Insurance Premium Prediction - Notebooks

This repository contains the Jupyter Notebooks used for building and evaluating statistical models to predict insurance premiums. The project is part of a larger initiative to analyze car insurance data and optimize marketing strategies. The notebook workflow covers various stages from data preparation to model evaluation.

## Table of Contents

1. [Overview](#overview)
2. [Data Preparation](#data-preparation)
3. [Model Building](#model-building)
4. [Model Evaluation](#model-evaluation)
5. [Feature Importance Analysis](#feature-importance-analysis)
6. [Model Interpretability](#model-interpretability)
7. [Requirements](#requirements)
8. [How to Run](#how-to-run)

## Overview

The goal of this project is to predict insurance premiums (TotalPremium) and claims (TotalClaims) based on a variety of features in the dataset, including customer details, vehicle information, and policy specifics. The project includes multiple models such as Linear Regression, Random Forest, and XGBoost, with a focus on feature importance and model interpretability.

## Data Preparation

The data preparation steps include:

- **Handling Missing Data:** Imputing or removing missing values based on their nature and quantity.
- **Feature Engineering:** Creating new features that might be relevant to the prediction of `TotalPremium` and `TotalClaims`.
- **Encoding Categorical Data:** Converting categorical variables into numeric formats using techniques like one-hot encoding or label encoding.
- **Train-Test Split:** Dividing the dataset into training and testing sets (typically 80%/20% split).

## Model Building

The following machine learning models are built and trained:

- **Linear Regression**
- **Random Forest**
- **XGBoost (Gradient Boosting Machine)**

Each model is evaluated on the training dataset and tested on the test dataset to assess performance.

## Model Evaluation

The models are evaluated using common metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **R-score**

These metrics help compare model performance and identify the most effective approach for predicting insurance premiums.

## Feature Importance Analysis

We perform feature importance analysis to understand which features are most influential in predicting the insurance premium. This analysis helps in identifying the key drivers of insurance premiums.

## Model Interpretability

We use **SHAP (SHapley Additive exPlanations)** and **LIME (Local Interpretable Model-agnostic Explanations)** to interpret the predictions of the models and understand how individual features influence the outcomes.

## Requirements

To run the notebooks, you will need to install the following libraries:

- pandas
- numpy
- scikit-learn
- shap
- matplotlib
- seaborn

You can install these dependencies using the following command:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Insurance-Premium-Prediction.git
   ```

2. Navigate to the `notebooks` directory:

   ```bash
   cd Insurance-Premium-Prediction/notebooks
   ```

3. Open the Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Run the notebooks in sequence to execute the data preparation, model building, evaluation, and interpretation steps.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The dataset used in this project is based on historical car insurance claim data.
- Thanks to the contributors of the open-source libraries used in this project.
