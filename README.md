# Price Predictor

## Overview

This project is designed to predict property prices using various machine learning models. The main components include model training, serving, and deployment with Docker. The application is structured to allow training of models, building Docker images, and deploying these models in a scalable environment.

## Project Structure

- `app.py`: The Flask application that serves the trained models.
- `dataProcessor.py`: Handles data loading and preprocessing.
- `modelHandler.py`: Manages model training and evaluation.
- `modelTrainer.py`: Contains the main script for training models.
- `requirements.txt`: Lists the dependencies required for the project.
- `Dockerfile`: Defines the Docker image build process for training and serving models.
- `.github/`: Contains GitHub Actions workflows for CI/CD.
- `first_5_rows.csv`: Sample data file for testing.
- `readme.md`: This file.
- `venv/`: Virtual environment directory (ignored in version control).

## Features

1. **Model Training**: 
   - The `ModelTrainer` class in `modelTrainer.py` is used for training and evaluating various models including SVR, RandomForest, and LinearRegression.

2. **Model Serving**:
   - The `app.py` file sets up a Flask server that serves the trained models. The Flask application listens for POST requests at the `/predict` endpoint to make predictions based on the input data.

3. **Docker Integration**:
   - **Dockerfile**: Defines multi-stage builds for training and serving models.
   - **Build Targets**:
     - `train`: Runs the training process and saves the best-performing model.
     - `pack-serving`: Builds a Docker image for serving the trained model.

4. **CI/CD Pipeline**:
   - **Training**: On each push to the `main` branch, the training process is triggered to ensure all models are trained and evaluated.
   - **Docker Build & Push**: On each tag creation, a Docker image is built and pushed to a public Docker registry.

## Getting Started

### Prerequisites

- Docker
- Git

### Building and Running Locally

1. **Build the Docker Image**

   ```bash
   docker build -t test-images --target pack-serving .
   docker run -p 5001:5001 test-images
    ```
   
### to test play with year to build and remodAdd to check up the prediction, in order to validate truthability there is a real data in first_5_rows.py
```curl -X POST http://127.0.0.1:5000/predict \
    -H "Content-Type: application/json" \
    -d '[
         {
           "MSSubClass": 60,
           "MSZoning": "RL",
           "LotArea": 8450,
           "LotConfig": "Inside",
           "BldgType": "1Fam",
           "OverallCond": 5,
           "YearBuilt": 1900,
           "YearRemodAdd": 2003,
           "Exterior1st": "VinylSd",
           "BsmtFinSF2": 0.0,
           "TotalBsmtSF": 856.0
         }
       ]'

```
### to see model performance

    ```
    curl http://127.0.0.1:5001/model_performance
    ```