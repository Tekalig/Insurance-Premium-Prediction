import pytest
import pandas as pd
from src.data_processing import load_data
from src.eda import summarize_data

def test_load_data():
    df = load_data(file_path)
    assert isinstance(df, pd.DataFrame), "Data should be loaded as a DataFrame"

def test_summarize_data():
    df = load_data('data/raw/sample_data.csv')
    summary = summarize_data(df)
    assert 'TotalPremium' in summary.columns, "'TotalPremium' should be a column in the summary"
