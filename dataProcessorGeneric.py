import os
import psutil
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sys
from sklearn.model_selection import train_test_split

class DataProcessorGeneric:
    def __init__(self, file_path=None, memory_threshold=85):
        self.file_path = file_path
        self.dataset = None
        self.object_cols = None
        self.df_final = None
        self.memory_threshold = memory_threshold

    def monitor_memory(self):
        """Check the current memory usage and terminate the process if it exceeds the threshold."""
        memory_usage = psutil.virtual_memory().percent
        if memory_usage > self.memory_threshold:
            print(f"Memory usage exceeded {self.memory_threshold}%. Terminating the process to avoid overload.")
            sys.exit(1)

    def load_data(self):
        """Load dataset from a local file using Pandas."""
        self.monitor_memory()
        if self.file_path:
            self.dataset = pd.read_csv(self.file_path)
        else:
            raise ValueError("No data source specified. Provide a local file path.")
        self.monitor_memory()
        return self.dataset

    def clean_data(self):
        """Clean the dataset by handling missing values and encoding categorical features using Label Encoding."""
        self.monitor_memory()
        # Drop rows with missing values
        self.dataset = self.dataset.dropna()

        # Identify categorical columns
        self.object_cols = [col for col in self.dataset.columns if self.dataset[col].dtype == 'object']

        # Label encode categorical columns
        label_encoders = {}
        for col in self.object_cols:
            le = LabelEncoder()
            self.dataset.loc[:, col] = le.fit_transform(self.dataset[col])
            label_encoders[col] = le

        self.df_final = self.dataset.copy()

        # Print the first 5 rows with all columns
        print("First 5 rows of the cleaned dataset:")
        print(self.df_final.head(5))

        self.monitor_memory()

        return self.df_final

    def save_clean_data(self, output_file='processed.csv'):
        """Save the cleaned dataset to a CSV file."""
        if self.df_final is not None:
            self.df_final.to_csv(output_file, index=False)
            print(f"Cleaned data saved to {output_file}")
        else:
            print("No cleaned data available to save.")

    def explore_data(self):
        """Explore the dataset to gain insights, such as identifying categorical and numerical columns."""
        self.monitor_memory()
        # Display the first 5 rows
        print("First 5 rows of the dataset:")
        print(self.dataset.head(5))

        # Show dataset shape
        print("\nDataset shape:", self.dataset.shape)

        # Identify categorical and numerical columns
        self.object_cols = [col for col in self.dataset.columns if self.dataset[col].dtype == 'object']
        num_cols = [col for col in self.dataset.columns if self.dataset[col].dtype in ['int64', 'float64']]

        print("\nCategorical variables:", len(self.object_cols))
        print("Numerical variables:", len(num_cols))
        self.monitor_memory()

    def split_data(self, columns=['rating']):
        #print(self.df_final['rating'])
        self.features = self.df_final.drop(columns=columns)# Handle missing values, scaling, etc.
        self.features.fillna(0, inplace=True)
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)

        # Splitting data into training and testing sets
        return train_test_split(
            self.features, self.df_final.get(columns), test_size=0.2, random_state=42
        )


def main():
    processor = DataProcessorGeneric(file_path='./data-sets/amazon_reviews.csv')

    # Run the data processing steps
    processor.load_data()
    processor.explore_data()
    processor.clean_data()

    # Save the cleaned data to a CSV file
    processor.save_clean_data()


if __name__ == '__main__':
    main()
