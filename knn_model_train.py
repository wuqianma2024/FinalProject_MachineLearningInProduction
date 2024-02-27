import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
from joblib import dump



def load_and_concatenate_csv(directory_pattern):
    """
    Load all CSV files matching the pattern from a directory and concatenate them into a single DataFrame.

    Parameters:
        directory_pattern (str): The pattern to match CSV files in the directory.

    Returns:
        DataFrame: Concatenated DataFrame containing data from all matched CSV files.
    """
    df_list = [pd.read_csv(file) for file in glob.glob(directory_pattern)]
    return pd.concat(df_list, ignore_index=True)

def encode_genres(df, column_name):
    """
    One-hot encode genres in a DataFrame column.

    Parameters:
        df (DataFrame): The DataFrame containing the column to be encoded.
        column_name (str): The name of the column containing genres.

    Returns:
        tuple: A tuple containing the DataFrame with an additional column for encoded genres and the MultiLabelBinarizer instance.
    """
    mlb = MultiLabelBinarizer()
    genres = df[column_name].apply(lambda x: x.split(',')).tolist()
    genres_encoded = mlb.fit_transform(genres)
    return df, genres_encoded, mlb

def train_knn_and_save(df, genres_encoded, mlb, knn_path='knn_model.joblib', mlb_path='mlb_instance.joblib'):
    """
    Train a K-Nearest Neighbors model based on encoded genres and save the model and MultiLabelBinarizer instance.

    Parameters:
        df (DataFrame): The DataFrame used for training.
        genres_encoded (ndarray): The numpy array containing one-hot encoded genres.
        mlb (MultiLabelBinarizer): The MultiLabelBinarizer instance used for encoding genres.
        knn_path (str): Path to save the trained KNN model.
        mlb_path (str): Path to save the MultiLabelBinarizer instance.
    """
    knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn.fit(genres_encoded)
    dump(knn, knn_path)
    dump(mlb, mlb_path)
    print(f"KNN model saved to {knn_path} and ML Binarizer saved to {mlb_path}.")




def main():
    # Load and concatenate CSV files
    df = load_and_concatenate_csv('datasets/books_*.csv')
    
    # Encode genres
    df, genres_encoded, mlb = encode_genres(df, 'Genres')
    
    # Train KNN model and save it along with the MultiLabelBinarizer instance
    train_knn_and_save(df, genres_encoded,mlb)
    
if __name__ == "__main__":
    main()

    