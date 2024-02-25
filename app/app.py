import streamlit as st
import pandas as pd
import numpy as np
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load
from sklearn.preprocessing import MultiLabelBinarizer

import os

# Ensure Streamlit uses the PORT environment variable in Heroku
port = int(os.environ.get("PORT", 8501))
os.environ["STREAMLIT_SERVER_PORT"] = str(port)



# Function to load and concatenate data from all matching CSV files
@st.cache_data
def load_data(directory_pattern):
    file_paths = glob.glob(directory_pattern)
    dfs = [pd.read_csv(file_path) for file_path in file_paths]
    return pd.concat(dfs, ignore_index=True)

@st.cache_data
def load_knn_model(model_path):
    model = load(model_path)
    return model

@st.cache_data
def load_mlb_instance(mlb_path):
    mlb_instance = load(mlb_path)
    return mlb_instance


# Initialize TF-IDF Vectorizer and fit-transform the corpus
def initialize_tfidf_vectorizer(corpus):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    return tfidf_vectorizer, tfidf_matrix

# Find similar books based on content
def find_similar_books_summary(user_input, tfidf_vectorizer, tfidf_matrix, df, top_n=5):
    user_input_tfidf = tfidf_vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(user_input_tfidf, tfidf_matrix).flatten()
    top_indices = np.argsort(cosine_similarities)[-top_n:][::-1]
    similar_books = df.iloc[top_indices]
    return similar_books

# Recommend books based on genre using pre-trained KNN and MultiLabelBinarizer
def recommend_books_genre(book_title, df, knn_model, mlb_instance, top_n=5):
    # Find the index of the book with the given title
    book_idx = df[df['Title'] == book_title].index[0]
    # Find genres of the given book
    book_genres = df.at[book_idx, 'Genres'].split(',')
    book_genres_encoded = mlb_instance.transform([book_genres])
    distances, indices = knn_model.kneighbors(book_genres_encoded, n_neighbors=top_n + 1)
    # Exclude the first match (the book itself) and fetch indices of similar books
    similar_books_indices = indices.flatten()[1:]
    similar_books = df.iloc[similar_books_indices]
    return similar_books

def display_similar_books(similar_books):
    """
    Display similar books in the Streamlit app.

    Parameters:
        similar_books (DataFrame): A DataFrame containing details of similar books.
    """
    for index, row in similar_books.iterrows():
        col1, col2 = st.columns([1, 3])  # Adjust column ratios if necessary
        with col1:
            st.image(row['cover_image'], width=100, use_column_width=True)
        with col2:
            st.write(f"**Title:** {row['Title']}")
            st.write(f"**Title:** {row['Genres']}")
            st.write(f"**Author:** {row['Author']}")
            st.write(f"**Year:** {row['Year']}")
        with st.expander("See Summary"):
            st.write(row['Summary'])
        st.markdown("---")  # Visual separator


def main():
    # Load data
    df = load_data('datasets/books_*.csv')
    
    # Load the saved KNN model and MultiLabelBinarizer instance
    knn_model = load_knn_model('knn_model.joblib')
    mlb_instance = load_mlb_instance('mlb_instance.joblib')
    
    # Initialize and transform corpus with TF-IDF
    tfidf_vectorizer, tfidf_matrix = initialize_tfidf_vectorizer(df['Summary'])
    
    # Streamlit UI for input
    st.title('Book Recommendation Engine')
    user_input = st.text_input('Describe the book you want to search:','A story about the impact of war on men, highlighting their struggles.')
    
    if user_input:
        similar_books = find_similar_books_summary(user_input, tfidf_vectorizer, tfidf_matrix, df)
        
        st.write("### Are you searching for books like:")
        # Display the similar books using the defined function
        display_similar_books(similar_books)

        # Use the title of the most similar book for genre-based recommendation
        most_similar_book_title = similar_books.iloc[0]['Title']
        st.write("## You might also like these books based on the genre of book you are searching:")
        genre_similar_books = recommend_books_genre(most_similar_book_title, df, knn_model, mlb_instance)
        # Display the similar books using the defined function
        display_similar_books(genre_similar_books)

if __name__ == "__main__":
    main()



