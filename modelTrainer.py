import joblib
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics
import numpy as np
from dataProcessor import DataProcessor
from dataProcessorGeneric import DataProcessorGeneric

DEFAULT_MODELS={
            'SVR': svm.SVR(),
            'RandomForest': RandomForestRegressor(random_state=42, n_estimators=10),
            'LinearRegression': LinearRegression()
        }

class ModelTrainer:
    def __init__(self, models = DEFAULT_MODELS, file_path=None, url = None):
        if url is not None:
            self.data_processor = DataProcessor(url)
        else:
            self.data_processor = DataProcessorGeneric(file_path)
        self.dataset = None
        self.df_final = None
        self.X_train = None
        self.X_valid = None
        self.Y_train = None
        self.Y_valid = None
        self.models = models

    def load_and_preprocess_data(self):
        self.dataset = self.data_processor.load_data()
        self.df_final = self.data_processor.clean_data()
        self.X_train, self.X_valid, self.Y_train, self.Y_valid = self.data_processor.split_data()

    def train_models(self):
        for name, model in self.models.items():
            print(f"Training {name}...")
            if name == "LogisticsRegression":
                self.Y_train = self.Y_train.values.ravel()
            model.fit(self.X_train, self.Y_train)

    def evaluate_models(self):
        scores = {}
        for name, model in self.models.items():
            Y_pred = model.predict(self.X_valid)
            rmse = np.sqrt(metrics.mean_squared_error(self.Y_valid, Y_pred))
            scores[name] = rmse
        print(scores)
        return scores

    def best_performer(self):
        res = {}
        scores = self.evaluate_models()
        best_model_name = min(scores, key=scores.get)
        res[best_model_name] = scores[best_model_name]
        return res
    ###method will predict possible popularity of a product , based on influencer feedback
    def recommend_top_5_products(self):
        # Step 1: Identify the best-performing model
        best_model_info = self.best_performer()
        best_model_name = next(iter(best_model_info))  # Get the name of the best model
        best_model = self.models.get(best_model_name)  # Retrieve the best model object

        # Step 2: Predict ratings using the best model
        predicted_ratings = best_model.predict(self.X_valid)

        # Step 3: Ensure X_valid is a DataFrame and has the correct index
        if not isinstance(self.X_valid, pd.DataFrame):
            recommendations = pd.DataFrame(self.X_valid)
        else:
            recommendations = self.X_valid.copy()

        # Step 4: Add predicted ratings to the recommendations DataFrame
        recommendations['predicted_rating'] = predicted_ratings

        # Step 5: Merge the original data with the recommendations to retain all columns
        # Reset indices to align them for merging
        recommendations.reset_index(drop=True, inplace=True)
        self.df_final.reset_index(drop=True, inplace=True)

        #  `self.df_final` contains the original product information
        #merged_recommendations = pd.concat([self.df_final, recommendations['predicted_rating']], axis=1)
        merged_recommendations = pd.DataFrame({'itemName': self.df_final['itemName'],'vote':self.df_final['vote'],'predicted_rating': recommendations['predicted_rating']})

        # Save the result to a CSV file -my debug
        #merged_recommendations.to_csv('predicted_candidate_decisions.csv', index=False)

        # Step 6: Group by `itemName` to aggregate predicted ratings by product
        grouped_recommendations = merged_recommendations.groupby('itemName').agg({
            'predicted_rating': 'mean',  # Aggregate predicted ratings
            #'userName': 'count',  # Optionally, count the number of ratings
            #'verified': 'first',  # Keep other columns as they are (you can choose how to aggregate)
            #'reviewText': 'first',
            'vote': 'sum'  # Sum the number of votes
        }).reset_index()

        # Step 7: Sort by predicted rating and number of votes
        sorted_recommendations = grouped_recommendations.sort_values(
            by=['predicted_rating', 'vote'],
            ascending=[False, False]
        )

        # Step 8: Return the top 5 products based on sorted DataFrame
        top_5_products = sorted_recommendations.head(5)

        return top_5_products


if __name__ == '__main__':
    # url = 'https://drive.google.com/file/d/1kqnB4J8FuF1k8xLIvfbPE1jsqUwd3wVH/view?usp=sharing'
    # trainer = ModelTrainer(url)
    # trainer.load_and_preprocess_data()
    # trainer.train_models()
    # performance = trainer.evaluate_models()
    #
    # print("Model Performance (RMSE):")
    # for model_name, rmse in performance.items():
    #     print(f"rmse-{model_name}: {rmse}")
    models = {
            'LogisticsRegression': LogisticRegression(random_state=42, max_iter=1000),
            'GradientBooster':  HistGradientBoostingClassifier(max_iter=100, random_state=42)
        }
    trainer = ModelTrainer(models=models, file_path='./data-sets/amazon_reviews.csv')
    trainer.load_and_preprocess_data()
    trainer.train_models()
    bp=trainer.best_performer()
    top_5_recommendations = trainer.recommend_top_5_products()
    #print(top_5_recommendations)
    print(top_5_recommendations.to_string(index=False))