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
```
curl -X POST http://127.0.0.1:5001/predict \
    -H "Content-Type: application/json" \
    -d '[
         {
           "MSSubClass": 60,
           "MSZoning": "RL",
           "LotArea": 8450,
           "LotConfig": "Inside",
           "BldgType": "1Fam",
           "OverallCond": 5,
           "YearBuilt": 2000,
           "YearRemodAdd": 2013,
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


### Added functionality - init FW for modeling
## Retrieving the 5 Most Popular Items Based on Influencer Feedback and Amazon Ratings

This project includes functionality to determine and return the top 5 most popular items, based on influencer feedback and how this feedback is rated by users on Amazon. The process involves training machine learning models on a dataset of Amazon reviews and using these models to predict the most popular products.

### Process Overview

1. **Data Processing**:
   - The data is first loaded and preprocessed using the `DataProcessor` or `DataProcessorGeneric` class. This step involves cleaning the data and splitting it into training and validation sets.

2. **Model Training**:
   - We train two machine learning models: `LogisticRegression` and `HistGradientBoostingClassifier`. These models are trained on the processed dataset to predict the popularity of items based on influencer feedback and associated ratings.

3. **Model Evaluation**:
   - After training, each model's performance is evaluated to identify the best-performing model based on the Root Mean Squared Error (RMSE).

4. **Top 5 Product Recommendations**:
   - The best-performing model is then used to predict the popularity ratings of products in the validation set. The results are aggregated, and the top 5 products are determined based on the predicted ratings and the number of votes each product received.

5. **API Endpoint**:
   - The top 5 most popular items can be retrieved via a simple HTTP GET request to the `/popular-goods` endpoint. The endpoint returns a JSON response containing the item names, predicted popularity ratings, and the number of votes.

### Example Usage

You can use the following cURL command to retrieve the top 5 most popular items:

```bash
curl http://127.0.0.1:5001/popular-goods
```

{
  "best_model": {
    "GradientBooster": 1.1208531794059018
  },
  "prediction": [
    {
      "itemName": "Frontier Co-Op Organic Catnip Leaf &amp; Flower, Cut &amp; Sifted, 1 Pound Bulk Bag",
      "predicted_rating": 5.0,
      "vote": 370
    },
    {
      "itemName": "Escape Proof Cat Harness with Leash Adjustable Soft Mesh - Best for Walking",
      "predicted_rating": 5.0,
      "vote": 287
    },
    {
      "itemName": "Jitterbug Flip Easy-to-Use Cell Phone for Seniors (Red) by GreatCall",
      "predicted_rating": 5.0,
      "vote": 217
    },
    {
      "itemName": "Drink Matcha Organic Green Tea Powder Set Bundle with Ceramic Tea Bowl, Handheld Electric Whisk and Bamboo Spoon, 16 oz.",
      "predicted_rating": 5.0,
      "vote": 166
    },
    {
      "itemName": "CINNAMON SMARTCAKE: Gluten Free, Sugar Free and Starch Free (8 x 2-packs)",
      "predicted_rating": 5.0,
      "vote": 122
    }
  ]
}


### Running the Docker container with the file path:
### When running the Docker container, pass the file path as an argument:

```bash
docker build -t my-model-trainer .
docker run -v /path/to/your/local/data:/app/data-sets my-model-trainer python modelTrainer.py 
```

### PLEASE NOTE:: './data-sets/amazon_reviews.csv' - default path if not supplied, file should be place there