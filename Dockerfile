# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Set the working directory in the Docker container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Adjust the datasets path if necessary and run the model training script before starting the app
# Ensure knn_model_train.py is executable and has the correct path to the datasets
RUN python app/knn_model_train.py

# Make port 8501 available to the world outside this container
#EXPOSE 8501

# Run the Streamlit app
# CMD ["streamlit", "run", "app/app.py"]
# Heroku assigns a dynamic port, so we do not EXPOSE a port here

# Run the Streamlit app, using the PORT environment variable provided by Heroku
CMD sh -c 'streamlit run --server.port $PORT app/app.py'
