import pandas as pd
from app import *  # Adjust import path as needed

def test_load_data():
    """
    Test the load_data function from the app module to ensure it correctly loads data from CSV files.
    """
    # Specify a path pattern for test CSV files.
    test_data_path = 'datasets/books_*.csv'
    
    # Load data from the specified CSV files.
    df = load_data(test_data_path)
    
    # Assert that the returned object is a pandas DataFrame and it's not empty.
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    
    # Verify that the DataFrame contains expected columns.
    expected_columns = ['Title', 'Author', 'Year', 'Genres', 'Summary', 'cover_image']
    assert all(column in df.columns for column in expected_columns)

    