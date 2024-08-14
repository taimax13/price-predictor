import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


class DataProcessor:
    def __init__(self, url):
        self.url = url
        self.dataset = None
        self.object_cols = None
        self.df_final = None
        self.X_train = None
        self.X_valid = None
        self.Y_train = None
        self.Y_valid = None

    def load_data(self):
        """Load dataset from Google Drive."""
        path = f'https://drive.google.com/uc?export=download&id={self.url.split("/")[-2]}'
        self.dataset = pd.read_csv(path).set_index('Id')
        return self.dataset

    def clean_data(self):
        """Clean the dataset by handling missing values and encoding categorical features."""
        # Drop rows with missing target values
        self.dataset = self.dataset.dropna(subset=['SalePrice'])

        # Identify categorical columns
        self.object_cols = [col for col in self.dataset.columns if self.dataset[col].dtype == 'object']

        # One-hot encode categorical columns
        OH_encoder = OneHotEncoder(sparse_output=False)
        OH_cols = pd.DataFrame(OH_encoder.fit_transform(self.dataset[self.object_cols]),
                               index=self.dataset.index,
                               columns=OH_encoder.get_feature_names_out())

        # Concatenate with numerical features
        df_final = self.dataset.drop(self.object_cols, axis=1)
        self.df_final = pd.concat([df_final, OH_cols], axis=1)
        return self.df_final

    def split_data(self):
        """Split the cleaned data into training and validation sets."""
        X = self.df_final.drop(['SalePrice'], axis=1)
        Y = self.df_final['SalePrice']
        self.X_train, self.X_valid, self.Y_train, self.Y_valid = train_test_split(
            X, Y, train_size=0.8, test_size=0.2, random_state=0)
        return self.X_train, self.X_valid, self.Y_train, self.Y_valid

    def analyze_dataset(self):
        """Perform analysis on the dataset."""
        # Display the first 5 rows of the dataset
        print("First 5 rows of the dataset:")
        print(self.dataset.head(5))

        # Display dataset shape
        print("\nDataset shape:", self.dataset.shape)

        # Identify and count categorical, integer, and float variables
        self.object_cols = [col for col in self.dataset.columns if self.dataset[col].dtype == 'object']
        num_cols = [col for col in self.dataset.columns if self.dataset[col].dtype in ['int', 'float']]

        print("\nCategorical variables:", len(self.object_cols))
        print("Integer variables:", len([col for col in num_cols if self.dataset[col].dtype == 'int']))
        print("Float variables:", len([col for col in num_cols if self.dataset[col].dtype == 'float']))

        # Plot correlation heatmap for numerical features
        plt.figure(figsize=(12, 6))
        sns.heatmap(self.dataset.select_dtypes(include='number').corr(), cmap='BrBG', fmt='.2f', linewidths=2,
                    annot=True)
        plt.title('Correlation Heatmap')
        plt.show()

        # Plot unique values of categorical features
        unique_values = [self.dataset[col].nunique() for col in self.object_cols]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=self.object_cols, y=unique_values)
        plt.title('No. Unique Values of Categorical Features')
        plt.xlabel('Categorical Features')
        plt.ylabel('Number of Unique Values')
        plt.xticks(rotation=90)
        plt.show()

        # Plot distribution of categorical features
        plt.figure(figsize=(12, 8))
        plt.suptitle('Categorical Features: Distribution')
        for index, col in enumerate(self.object_cols, start=1):
            plt.subplot((len(self.object_cols) + 1) // 2, 2, index)
            sns.barplot(x=self.dataset[col].value_counts().index, y=self.dataset[col].value_counts())
            plt.title(col)
            plt.xlabel('Categories')
            plt.ylabel('Counts')
            plt.xticks(rotation=90)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

# Example usage:
# processor = DataProcessor('your_google_drive_id')
# processor.load_data()
# processor.analyze_dataset()

