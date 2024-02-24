import streamlit as st
import pandas as pd
import random
import requests

# Load your dataset
df = pd.read_csv('books_dataset_with_images.csv')

# Streamlit app
def book_recommender():
    st.title('Book Recommender System')

    # Fetch session state
    session_state = st.session_state

    # Sidebar for user input
    st.sidebar.header('User Input')

    # Get user input for book search
    search_option = st.sidebar.selectbox('Search by:', ['Author', 'Title', 'Content'])
    user_input = st.sidebar.text_input(f'Enter {search_option.lower()}:')
    user_input = user_input.title()  # Convert input to title case for case-insensitivity

    # Fetch images for the initially displayed random books
    if 'random_books' not in session_state:
        session_state.random_books = random.sample(range(len(df)), min(9, len(df)))

    # Display the title screen with a fixed 3x3 grid
    st.subheader('Featured Books')

    # Create a fixed 3x3 grid layout
    for i in range(3):
        columns = st.columns(3)  # 3 books per row
        for j in range(3):
            index = i * 3 + j
            book = df.iloc[session_state.random_books[index]]
            with columns[j]:
                # Clickable box with book information and the fetched image
                if st.button(f"**{book['Title']}** by {book['Author']}"):
                    # Update the selected book in session state
                    session_state.selected_book = book

                st.image(book['cover_image'], use_column_width=True)  # Use the 'cover_image' column
                st.write(f"Summary: {book['Summary'][:150]}...")  # Crop to 150 characters
                st.markdown("---")  # Add a separator between books

    # Display selected book details if available
    if 'selected_book' in session_state and session_state.selected_book is not None:
        st.subheader('Selected Book Details')
        selected_book = session_state.selected_book
        st.image(selected_book['cover_image'], use_column_width=True)  # Use the 'cover_image' column
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
            # Clear the front page and display recommended books
            st.subheader('Recommended Books')
            columns = st.columns(3)  # 3 books per row
            for index, book in result.iterrows():
                with columns[index % 3]:
                    # Clickable box with book information and the fetched image
                    if st.button(f"**{book['Title']}** by {book['Author']}"):
                        # Update the selected book in session state
                        session_state.selected_book = book

                    st.image(book['cover_image'], use_column_width=True)  # Use the 'cover_image' column
                    st.write(f"Summary: {book['Summary'][:150]}...")  # Crop to 150 characters
                    st.markdown("---")  # Add a separator between books
        else:
            st.warning('No matching books found.')

if __name__ == '__main__':
    book_recommender()
