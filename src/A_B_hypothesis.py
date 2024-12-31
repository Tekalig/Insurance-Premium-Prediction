# Description: This script contains functions to test hypotheses related to risk differences and margin differences.s
import pandas as pd
from scipy import stats


# Hypothesis 1: "There are no risk differences across provinces."
# Null Hypothesis (H₀): There are no significant risk differences across provinces.
# Alternative Hypothesis (H₁): There are significant risk differences across provinces.

def test_risk_difference_across_provinces(data):
    """
    Conduct a chi-square test to determine if there are significant risk differences across provinces.

    Parameters:
    data (DataFrame): The dataset containing insurance policy data.

    Returns:
    p_value (float): p-value of the chi-square test.
    """
    # Grouping by province and calculating the sum of claims (risk) in each province
    province_claims = data.groupby('Province')['TotalClaims'].sum().reset_index()

    # Creating contingency table (number of claims in each province)
    contingency_table = pd.crosstab(province_claims['Province'], province_claims['TotalClaims'])

    # Perform Chi-Square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    return p_value


# Hypothesis 2: "There are no risk differences between zip codes."
# Null Hypothesis (H₀): There are no significant risk differences between zip codes.
# Alternative Hypothesis (H₁): There are significant risk differences between zip codes.

def test_risk_difference_between_zip_codes(data):
    """
    Conduct a chi-square test to determine if there are significant risk differences between zip codes.

    Parameters:
    data (DataFrame): The dataset containing insurance policy data.

    Returns:
    p_value (float): p-value of the chi-square test.
    """
    # Grouping by zip code and calculating the sum of claims (risk) in each zip code
    zip_claims = data.groupby('PostalCode')['TotalClaims'].sum().reset_index()

    # Creating contingency table (number of claims in each zip code)
    contingency_table = pd.crosstab(zip_claims['PostalCode'], zip_claims['TotalClaims'])

    # Perform Chi-Square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    return p_value


# Hypothesis 3: "There are no significant margin (profit) differences between zip codes."
# Null Hypothesis (H₀): There are no significant margin (profit) differences between zip codes.
# Alternative Hypothesis (H₁): There are significant margin (profit) differences between zip codes.

def test_margin_difference_between_zip_codes(data):
    """
    Conduct a T-test to determine if there are significant margin (profit) differences between zip codes.

    Parameters:
    data (DataFrame): The dataset containing insurance policy data.

    Returns:
    p_value (float): p-value of the T-test.
    """
    # Calculating margin (profit) for each record
    data['ProfitMargin'] = data['TotalPremium'] - data['TotalClaims']

    # Grouping by zip code and calculating the mean margin in each zip code
    zip_margin = data.groupby('PostalCode')['ProfitMargin'].mean().reset_index()

    # Performing a T-test between two zip codes (example: 12345 and 67890)
    zip_a = zip_margin[zip_margin['PostalCode'] == 9870]['ProfitMargin'] + 1
    zip_b = zip_margin[zip_margin['PostalCode'] == 9869]['ProfitMargin'] + 1

    # Perform T-test
    t_stat, p_value = stats.ttest_ind(zip_a, zip_b)

    return p_value


# Hypothesis 4: "There are no significant risk differences between Women and Men."
# Null Hypothesis (H₀): There are no significant risk differences between Women and Men.
# Alternative Hypothesis (H₁): There are significant risk differences between Women and Men.

def test_risk_difference_by_gender(data):
    """
    Conduct a T-test to determine if there are significant risk differences between women and men.

    Parameters:
    data (DataFrame): The dataset containing insurance policy data.

    Returns:
    p_value (float): p-value of the T-test.
    """
    # Grouping by gender and calculating the total claims (risk) for each gender
    women_claims = data[data['Gender'] == 'Female']['TotalClaims']
    men_claims = data[data['Gender'] == 'Male']['TotalClaims']

    # Perform T-test
    t_stat, p_value = stats.ttest_ind(women_claims, men_claims)

    return p_value