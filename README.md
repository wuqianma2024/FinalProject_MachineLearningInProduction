
# Final Project: Machine Learning in Production

## Overview
This project demonstrates the end-to-end process of deploying a machine learning model into production. Using Streamlit for the web interface and Heroku for hosting, it showcases how to operationalize a machine learning model, making it accessible via a web application.

## Features
- **Machine Learning Model**: Utilizes a pre-trained model to perform predictions. (Specify the model and its use case)
- **Streamlit Application**: Interactive web interface for model interaction.
- **Docker Integration**: Containerized application deployment for consistency across development and production environments.
- **CI/CD Pipeline**: Automated deployment pipeline using GitHub Actions, ensuring that the latest version of the app is always available.
- **Heroku Hosting**: Leveraging Heroku's platform for deploying and hosting the application.

## Installation
To set up the project locally, follow these steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/wuqianma2024/FinalProject_MachineLearningInProduction.git
   cd FinalProject_MachineLearningInProduction



2. **Install dependencies**

    ```bash
    pip install -r requirements.txt

3. **Run test**
    '''bash
    pytest


4.  **Run the Streamlit app**

    ```bash
    streamlit run app.py



## docker image

    '''bash
    docker pull ghcr.io/wuqianma2024/books_recommend:latest

## Usage
After installation, the Streamlit app can be accessed locally by navigating to http://localhost:8501 in your web browser. Interact with the machine learning model by inputting data into the provided fields and submitting it for prediction.

## Deployment
This project is set up for continuous deployment using GitHub Actions and Heroku:

GitHub Actions automates the testing and deployment of the Docker container to Heroku upon every commit to the main branch.
Heroku hosts the deployed application, accessible via a public URL.
For detailed steps on setting up the CI/CD pipeline and deploying to Heroku, refer to the heroku-deploy.yml file in the .github/workflows directory.

## Contributing
Contributions to improve the application are welcome. To contribute:

Fork the repository.
Create a new branch for your feature (git checkout -b feature/AmazingFeature).
Commit your changes (git commit -m 'Add some AmazingFeature').
Push to the branch (git push origin feature/AmazingFeature).
Open a pull request.
## License
This project is licensed under the MIT License - see the LICENSE file for details.
