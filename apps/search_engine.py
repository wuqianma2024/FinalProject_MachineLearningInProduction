import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Function to initialize TF-IDF Vectorizer
def initialize_tfidf_vectorizer(corpus):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    return tfidf_vectorizer, tfidf_matrix

# Function to find similar books
def find_similar_books(user_input, tfidf_vectorizer, tfidf_matrix, df, top_n=5):
    user_input_tfidf = tfidf_vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(user_input_tfidf, tfidf_matrix).flatten()
    top_indices = np.argsort(cosine_similarities)[-top_n:][::-1]
    similar_books = df.iloc[top_indices]
    return similar_books

def main():
    # Load data
    @st.cache_data  # Cache the loaded data to avoid reloading on every Streamlit run
    def load_data(file_path):
        return pd.read_csv(file_path)

    # Load and preprocess data
    df = load_data('books_1.csv')
    tfidf_vectorizer, tfidf_matrix = initialize_tfidf_vectorizer(df['Summary'])

    # Streamlit UI
    st.title('Book Search Engine')
    user_input = st.text_input('Enter your search query:', 'A story about the impact of war on soldiers, highlighting their struggles.')

    if user_input:
        # Find similar books based on user input
        similar_books = find_similar_books(user_input, tfidf_vectorizer, tfidf_matrix, df)
        
        # Display similar books
        for index, row in similar_books.iterrows():
            col1, col2 = st.columns(2)
            with col1:
                st.image(row['cover_image'], width=150)

            with col2:
                st.write(f"**Book:** \n{row['Title']}\n"
                        f"---\n"  # Add a line break
                        f"**Author:** \n{row['Author']}\n"
                        f"---\n"  # Add a line break
                        f"**Year:** \n{row['Year']}\n"
                        f"---\n")  # Add a line break
                
                
            with st.expander(f"**Summary**"):
                    st.write(row['Summary'])
            st.write("________________________")

if __name__ == "__main__":
    main()



