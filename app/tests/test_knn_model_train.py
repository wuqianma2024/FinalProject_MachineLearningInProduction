from unittest.mock import patch
import pandas as pd
import numpy as np
from knn_model_train import load_and_concatenate_csv, encode_genres, MultiLabelBinarizer,train_knn_and_save



@patch('glob.glob')
@patch('pandas.read_csv')
def test_load_and_concatenate_csv(mock_read_csv, mock_glob):
    mock_glob.return_value = ['dummy_path.csv']
    mock_df = pd.DataFrame({'dummy_column': [1, 2, 3]})
    mock_read_csv.return_value = mock_df

    result = load_and_concatenate_csv('dummy_pattern')
    mock_glob.assert_called_once_with('dummy_pattern')
    assert not result.empty




def test_encode_genres():
    df = pd.DataFrame({'Genres': ['Sci-Fi,Adventure', 'Drama']})
    df, encoded, mlb = encode_genres(df, 'Genres')
    assert encoded.shape == (2, len(mlb.classes_))
    assert 'Sci-Fi' in mlb.classes_


@patch('joblib.dump')
def test_train_knn_and_save(mock_dump):
    df = pd.DataFrame({'Genres': ['Sci-Fi,Adventure', 'Drama']})
    genres_encoded = np.array([[1, 0], [0, 1]])
    mlb = MultiLabelBinarizer().fit(df['Genres'].apply(lambda x: x.split(',')))

    train_knn_and_save(df, genres_encoded, mlb)
    assert mock_dump.call_count == 0

