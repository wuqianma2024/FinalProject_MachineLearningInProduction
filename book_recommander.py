import streamlit as st
import pandas as pd
import random
import requests

# Load your dataset
df = pd.read_csv('books_dataset.csv')

# Function to get book cover from Google Books API
def get_google_books_cover(title, author):
    base_url = "https://www.googleapis.com/books/v1/volumes"
    params = {
        'q': f"{title}+inauthor:{author}",
        'maxResults': 1
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'items' in data and data['items']:
            thumbnail_url = data['items'][0]['volumeInfo'].get('imageLinks', {}).get('thumbnail')
            if thumbnail_url:
                return thumbnail_url
    return "https://via.placeholder.com/150"  # Placeholder image if cover not found

# Function to create a unique session state
def get_session():
    session = st.session_state
    if not hasattr(session, "selected_book"):
        session.selected_book = None
    return session

# Streamlit app
def book_recommender():
    st.title('Book Recommender System')

    # Fetch session state
    session_state = get_session()

    # Sidebar for user input
    st.sidebar.header('User Input')

    # Get user input for book search
    search_option = st.sidebar.selectbox('Search by:', ['Author', 'Title', 'Content'])
    user_input = st.sidebar.text_input(f'Enter {search_option.lower()}:')
    user_input = user_input.title()  # Convert input to title case for case-insensitivity

    # Fetch images for the initially displayed random books
    random_books = random.sample(range(len(df)), min(9, len(df)))
    df['Image_URL'] = df.iloc[random_books].apply(lambda row: get_google_books_cover(row['Title'], row['Author']), axis=1)

    # Display the title screen with a fixed 3x3 grid
    st.subheader('Featured Books')

    # Create a fixed 3x3 grid layout
    for i in range(3):
        columns = st.columns(3)  # 3 books per row
        for j in range(3):
            index = i * 3 + j
            book = df.iloc[random_books[index]]
            with columns[j]:
                # Clickable box with book information and the fetched image
                if st.button(f"**{book['Title']}** by {book['Author']}"):
                    # Update the selected book in session state
                    session_state.selected_book = book

                st.image(book['Image_URL'], use_column_width=True)
                st.write(f"Summary: {book['Summary'][:150]}...")  # Crop to 150 characters
                st.markdown("---")  # Add a separator between books

    # Display selected book details if available
    if session_state.selected_book:
        st.subheader('Selected Book Details')
        selected_book = session_state.selected_book
        st.image(selected_book['Image_URL'], use_column_width=True)
        st.title(f"**{selected_book['Title']}** by {selected_book['Author']}")
        st.write("Full Summary:")
        st.write(selected_book['Summary'])
        
        # Close button to clear the selection
        if st.button("Close"):
            session_state.selected_book = None

    # Sidebar for user input
    st.sidebar.header('User Input')

    # Recommender logic
    if user_input:
        # Case-insensitive search based on user-selected option
        if search_option == 'Author':
            result = df[df['Author'].str.contains(user_input, case=False)]
        elif search_option == 'Title':
            result = df[df['Title'].str.contains(user_input, case=False)]
        else:  # search_option == 'Content'
            result = df[df['Summary'].str.contains(user_input, case=False)]

        if not result.empty:
            # Fetch images for the recommended books
            result['Image_URL'] = result.apply(lambda row: get_google_books_cover(row['Title'], row['Author']), axis=1)

            # Clear the front page and display recommended books
            st.subheader('Recommended Books')
            columns = st.columns(3)  # 3 books per row
            for index, book in result.iterrows():
                with columns[index % 3]:
                    # Clickable box with book information and the fetched image
                    if st.button(f"**{book['Title']}** by {book['Author']}"):
                        # Update the selected book in session state
                        session_state.selected_book = book

                    st.image(book['Image_URL'], use_column_width=True)
                    st.write(f"Summary: {book['Summary'][:150]}...")  # Crop to 150 characters
                    st.markdown("---")  # Add a separator between books
        else:
            st.warning('No matching books found.')

if __name__ == '__main__':
    book_recommender()
