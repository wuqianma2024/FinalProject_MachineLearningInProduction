# Import necessary functions and classes from app and knn_model_train modules.
from app import load_data, initialize_tfidf_vectorizer, find_similar_books_summary, recommend_books_genre
from knn_model_train import load_and_concatenate_csv, encode_genres, train_knn_and_save, NearestNeighbors

def test_app_flow():
    # Load dataset from CSV files matching the pattern 'datasets/books_*.csv'.
    df = load_data('datasets/books_*.csv')  
    
    # Initialize the TF-IDF vectorizer with the summaries from the loaded dataset, then transform the summaries into a TF-IDF matrix.
    tfidf_vectorizer, tfidf_matrix = initialize_tfidf_vectorizer(df['Summary'])
    
    # Find books similar to a user-provided summary about the impact of war using the TF-IDF matrix.
    user_input = "A story about the impact of war"
    similar_books = find_similar_books_summary(user_input, tfidf_vectorizer, tfidf_matrix, df)
    
    # Verify that the function returns a non-empty DataFrame of similar books.
    assert not similar_books.empty

def test_model_training_flow():
    # Load and concatenate all CSV files in the specified directory into a single DataFrame.
    df = load_and_concatenate_csv('datasets/books_*.csv')
    
    # Encode the 'Genres' column into a format suitable for machine learning models.
    df, genres_encoded, mlb = encode_genres(df, 'Genres')
    
    # Initialize and train a K-Nearest Neighbors (KNN) model using the cosine similarity metric.
    knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn.fit(genres_encoded)
    
    # Test the KNN model by finding the 2 nearest neighbors of the first genre encoded row.
    test_genres_encoded = genres_encoded[0:1]  # Use the first row as a test instance.
    distances, indices = knn.kneighbors(test_genres_encoded, n_neighbors=2)
    
    # Ensure that exactly 2 neighbors are found as expected.
    assert len(indices[0]) == 2
    