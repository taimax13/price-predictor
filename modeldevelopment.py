import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from dataProcessorGeneric import DataProcessorGeneric


class ModelDevelopmentClass:
    def __init__(self, df):
        self.df = df
        self.features = df.drop(columns=['rating'])
        self.target = df['rating']
        self.models = {}
        self.best_model = None

    def preprocess_data(self):
        # Handle missing values, scaling, etc.
        self.features.fillna(0, inplace=True)
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)

        # Splitting data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42
        )

    def build_models(self):
        # Building a Random Forest Classifier
        # rf_model = RandomForestClassifier(max_depth=10,random_state=42, n_jobs=-1)
        # rf_model.fit(self.X_train, self.y_train)
        hgb_model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
        hgb_model.fit(self.X_train, self.y_train)
        self.models['GradientBooster'] = hgb_model

        # Building a Logistic Regression model
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(self.X_train, self.y_train)
        self.models['LogisticRegression'] = lr_model


    def run(self):
        self.preprocess_data()
        self.build_models()
        self.evaluate_models()

def main():
    processor = DataProcessorGeneric(file_path='./data-sets/amazon_reviews.csv')

    # Run the data processing steps
    processor.load_data()
    processor.explore_data()
    df = processor.clean_data()
    model_dev = ModelDevelopmentClass(df)
    model_dev.run()


if __name__ == '__main__':
    main()
