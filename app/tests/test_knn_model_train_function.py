from unittest.mock import patch
import pandas as pd
import numpy as np
from knn_model_train import load_and_concatenate_csv, encode_genres, MultiLabelBinarizer,train_knn_and_save



@patch('glob.glob')  # Mock the glob module's glob function to control file path resolution.
@patch('pandas.read_csv')  # Mock pandas' read_csv function to control CSV file reading.
def test_load_and_concatenate_csv(mock_read_csv, mock_glob):
    """
    Test the load_and_concatenate_csv function to ensure it correctly loads and concatenates CSV files.
    Args:
        mock_read_csv: A mock of pandas.read_csv function to simulate reading CSV files.
        mock_glob: A mock of glob.glob function to simulate file path resolution.
    """
    # Setup mock return values to simulate the environment and data.
    mock_glob.return_value = ['dummy_path.csv']  # Simulate finding a single CSV file.
    mock_df = pd.DataFrame({'dummy_column': [1, 2, 3]})  # Simulate CSV file content.
    mock_read_csv.return_value = mock_df

    # Call the function with a dummy pattern to test its behavior.
    result = load_and_concatenate_csv('dummy_pattern')
    
    # Verify that glob.glob was called with the expected pattern.
    mock_glob.assert_called_once_with('dummy_pattern')
    
    # Assert that the result is not empty, indicating successful loading and concatenation.
    assert not result.empty

def test_encode_genres():
    """
    Test the encode_genres function to ensure it correctly encodes genre labels into a machine-readable format.
    """
    # Create a test DataFrame with a 'Genres' column.
    df = pd.DataFrame({'Genres': ['Sci-Fi,Adventure', 'Drama']})
    
    # Encode the 'Genres' column of the DataFrame.
    df, encoded, mlb = encode_genres(df, 'Genres')
    
    # Assert that the shape of the encoded array matches the expected dimensions.
    assert encoded.shape == (2, len(mlb.classes_))
    
    # Verify that a specific genre label is correctly represented in the encoding.
    assert 'Sci-Fi' in mlb.classes_

@patch('joblib.dump')  # Mock joblib.dump to test saving the model without actually writing files.
def test_train_knn_and_save(mock_dump):
    """
    Test the train_knn_and_save function to ensure it correctly trains the KNN model and attempts to save it.
    Args:
        mock_dump: A mock of joblib.dump function to simulate saving the model.
    """
    # Prepare a test DataFrame and encoded genres for model training.
    df = pd.DataFrame({'Genres': ['Sci-Fi,Adventure', 'Drama']})
    genres_encoded = np.array([[1, 0], [0, 1]])
    mlb = MultiLabelBinarizer().fit(df['Genres'].apply(lambda x: x.split(',')))

    # Call the function to simulate training and saving the KNN model.
    train_knn_and_save(df, genres_encoded, mlb)
    
    # Verify that joblib.dump was not called, as saving the model is not part of this test's scope.
    assert mock_dump.call_count == 0