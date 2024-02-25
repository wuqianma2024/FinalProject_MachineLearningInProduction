import pandas as pd
from app import *  # Adjust import path as needed

def test_load_data():
    """Test that CSV data is loaded correctly."""
    test_data_path = 'datasets/books_*.csv'  # Use a test CSV file
    df = load_data(test_data_path)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    expected_columns = ['Title', 'Author', 'Year', 'Genres', 'Summary', 'cover_image']
    assert all(column in df.columns for column in expected_columns)


