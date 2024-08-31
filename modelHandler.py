import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from modelTrainer import ModelTrainer


class ModelHandler:
    def __init__(self, url):
        self.trainer = ModelTrainer(url=url)
        self.trainer.load_and_preprocess_data()
        self.trainer.train_models()
        self.models = self.trainer.models
        self.X_valid = self.trainer.X_valid
        self.Y_valid = self.trainer.Y_valid

        # Store categorical columns for preprocessing
        self.object_cols = self.trainer.data_processor.dataset.select_dtypes(include='object').columns.tolist()
        self.OH_encoder = OneHotEncoder(sparse_output=False)
        self.OH_encoder.fit(self.trainer.data_processor.dataset[self.object_cols])

    def preprocess_input(self, input_df):
        # Preprocess input data similarly to how training data was preprocessed
        OH_cols = pd.DataFrame(self.OH_encoder.transform(input_df[self.object_cols]))
        OH_cols.index = input_df.index
        OH_cols.columns = self.OH_encoder.get_feature_names_out()
        return pd.concat([input_df.drop(self.object_cols, axis=1), OH_cols], axis=1)

    def predict(self, input_df):
        input_final = self.preprocess_input(input_df)
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(input_final).tolist()
        return predictions

    def evaluate_models(self):
        scores = self.trainer.evaluate_models()
        return scores
