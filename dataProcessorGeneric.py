import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import psutil
import sys

class DataProcessorGeneric:
    def __init__(self, file_path=None, memory_threshold=85):
        self.file_path = file_path
        self.dataset = None
        self.object_cols = None
        self.df_final = None
        self.memory_threshold = memory_threshold
        self.label_encoders = {}  # Store LabelEncoders here

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

        # Convert boolean columns to integers
        bool_cols = self.dataset.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            self.dataset[col] = self.dataset[col].astype(int)

        # Identify categorical columns
        self.object_cols = [col for col in self.dataset.columns if self.dataset[col].dtype == 'object']

        # Label encode categorical columns
        for col in self.object_cols:
            le = LabelEncoder()
            self.dataset[col] = le.fit_transform(self.dataset[col])
            self.label_encoders[col] = le  # Store the encoder

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

    def split_data(self, target_column='rating'):
        """Split the dataset into training and testing sets."""
        self.monitor_memory()
        features = self.df_final.drop(columns=[target_column])
        target = self.df_final[target_column]

        # Handle missing values, scaling, etc.
        features.fillna(0, inplace=True)
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # Splitting data into training and testing sets
        return train_test_split(
            features, target, test_size=0.2, random_state=42
        )

    def decode_columns(self, encoded_df):
        """Decode the label encoded columns back to their original values."""
        decoded_df = encoded_df.copy()
        for col, le in self.label_encoders.items():
            decoded_df[col] = le.inverse_transform(encoded_df[col])
        return decoded_df


def main():
    processor = DataProcessorGeneric(file_path='./data-sets/amazon_reviews.csv')
    processor.load_data()
    processor.explore_data()
    processor.clean_data()
    processor.save_clean_data()
    predicted_df = processor.df_final.head(5)
    decoded_df = processor.decode_columns(predicted_df)
    print("Decoded DataFrame:")
    print(decoded_df)


if __name__ == '__main__':
    main()
