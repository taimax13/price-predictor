import joblib
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from dataProcessor import DataProcessor


class ModelTrainer:
    def __init__(self, url):
        self.data_processor = DataProcessor(url)
        self.dataset = None
        self.df_final = None
        self.X_train = None
        self.X_valid = None
        self.Y_train = None
        self.Y_valid = None
        self.models = {
            'SVR': svm.SVR(),
            'RandomForest': RandomForestRegressor(random_state=42, n_estimators=10),
            'LinearRegression': LinearRegression()
        }

    def load_and_preprocess_data(self):
        self.dataset = self.data_processor.load_data()
        self.df_final = self.data_processor.clean_data()
        self.X_train, self.X_valid, self.Y_train, self.Y_valid = self.data_processor.split_data()

    def train_models(self):
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.Y_train)

    def evaluate_models(self):
        scores = {}
        for name, model in self.models.items():
            Y_pred = model.predict(self.X_valid)
            rmse = np.sqrt(metrics.mean_squared_error(self.Y_valid, Y_pred))
            scores[name] = rmse
        return scores


if __name__ == '__main__':
    url = 'https://drive.google.com/file/d/1kqnB4J8FuF1k8xLIvfbPE1jsqUwd3wVH/view?usp=sharing'
    trainer = ModelTrainer(url)
    trainer.load_and_preprocess_data()
    trainer.train_models()
    performance = trainer.evaluate_models()

    print("Model Performance (RMSE):")
    for model_name, rmse in performance.items():
        print(f"rmse-{model_name}: {rmse}")
