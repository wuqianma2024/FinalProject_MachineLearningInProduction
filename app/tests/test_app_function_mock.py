from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from app import initialize_tfidf_vectorizer,find_similar_books_summary,recommend_books_genre



def test_initialize_tfidf_vectorizer():
    corpus = ["book one summary", "book two summary"]
    vectorizer, matrix = initialize_tfidf_vectorizer(corpus)
    
    assert vectorizer is not None
    assert matrix.shape == (2, len(set(" ".join(corpus).split())))




@patch('app.cosine_similarity')
@patch('sklearn.feature_extraction.text.TfidfVectorizer.transform')
def test_find_similar_books_summary(mock_transform, mock_cosine_similarity):
    mock_transform.return_value = MagicMock()
    mock_cosine_similarity.return_value = np.array([[0.1, 0.9, 0.8, 0.7, 0.6]])
    df = pd.DataFrame({
        'Title': ['Book 1', 'Book 2', 'Book 3', 'Book 4', 'Book 5'],
        'Summary': ['Summary 1', 'Summary 2', 'Summary 3', 'Summary 4', 'Summary 5']
    })
    
    tfidf_vectorizer = MagicMock()
    tfidf_matrix = MagicMock()
    
    similar_books = find_similar_books_summary("book summary", tfidf_vectorizer, tfidf_matrix, df, top_n=3)
    
    assert len(similar_books) == 3
    assert similar_books.iloc[0]['Title'] == 'Book 2'





@patch('app.load_knn_model')
@patch('app.load_mlb_instance')
def test_recommend_books_genre(mock_load_mlb_instance, mock_load_knn_model):
    mock_knn_model = MagicMock()
    mock_knn_model.kneighbors.return_value = (np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]), np.array([[0, 1, 2, 3, 4]]))
    mock_load_knn_model.return_value = mock_knn_model
    
    mock_mlb_instance = MagicMock()
    mock_mlb_instance.transform.return_value = np.array([[1, 0, 1]])
    mock_load_mlb_instance.return_value = mock_mlb_instance
    
    df = pd.DataFrame({
        'Title': ['Book 1', 'Book 2', 'Book 3', 'Book 4', 'Book 5'],
        'Genres': ['Genre1', 'Genre2', 'Genre3', 'Genre1, Genre2', 'Genre3']
    })
    
    similar_books = recommend_books_genre("Book 1", df, mock_knn_model, mock_mlb_instance)
    
    # because the recommend book should not consider the reference book
    assert len(similar_books) == 4  # Adjust based on the logic 