import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class DataVisualizer:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.print_column_names()
        self.print_initial_data()

    def print_column_names(self):
        """Prints all column names in the dataset."""
        print("Column names in the dataset:")
        print(self.df.columns.tolist())

    def print_initial_data(self):
        """Prints a more concise and useful view of the first few rows and columns."""
        pd.set_option('display.max_columns', None)  # Ensure all columns are displayed

        print("\nFirst 5 rows of the dataset:")
        print(self.df.head())  # Print the first 5 rows of the dataset

    def preprocess_data(self):
        # Example preprocessing: categorize sentiment
        if 'rating' in self.df.columns:
            self.df['sentiment'] = self.df['rating'].apply(
                lambda x: 'Positive' if x >= 4 else ('Negative' if x <= 2 else 'Neutral'))
        else:
            print("Rating column not found.")

    def plot_sentiment_by_category(self, category_column='category', rating_column='rating'):
        if category_column in self.df.columns and 'sentiment' in self.df.columns:
            sentiment_by_category = self.df.groupby([category_column, 'sentiment']).size().unstack()
            print("Sentiment by category:")
            print(sentiment_by_category)  # Check the grouped data

            if not sentiment_by_category.empty:
                #plt.figure(figsize=(12, 8))
                sentiment_by_category.plot(kind='bar', stacked=True, color=['red', 'gray', 'green'], alpha=0.7)
                plt.title('Sentiment Distribution by Product Category')
                plt.xlabel('Product Category')
                plt.ylabel('Number of Reviews')
                plt.xticks(rotation=45, ha='right')
                plt.legend(title='Sentiment')
                plt.tight_layout()
                plt.show()
            else:
                print("No data to plot for sentiment by category.")
        else:
            print(f"{category_column} or 'sentiment' column not found.")

    def plot_helpful_votes_distribution(self, helpful_votes_column='helpful_votes'):
        if helpful_votes_column in self.df.columns:
            # Ensure the column is numeric and handle non-numeric data
            try:
                helpful_votes = pd.to_numeric(self.df[helpful_votes_column], errors='coerce')

                # Drop NaN values that resulted from conversion errors
                helpful_votes = helpful_votes.dropna()

                if not helpful_votes.empty:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(helpful_votes, bins=50, kde=True, color='blue')
                    plt.title('Distribution of Helpful Votes')
                    plt.xlabel('Helpful Votes')
                    plt.ylabel('Frequency')
                    plt.show()
                else:
                    print(f"No valid data to plot in column '{helpful_votes_column}'.")
            except Exception as e:
                print(f"An error occurred while processing the column '{helpful_votes_column}': {e}")
        else:
            print(f"'{helpful_votes_column}' column not found.")

    def plot_rating_discrepancies(self, sentiment_column='sentiment', rating_column='rating'):
        if sentiment_column in self.df.columns and rating_column in self.df.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=sentiment_column, y=rating_column, data=self.df, hue=sentiment_column, palette='Set3',
                        legend=False)
            plt.title('Star Rating Distribution by Text Sentiment')
            plt.xlabel('Text Sentiment')
            plt.ylabel('Star Rating')
            plt.show()
        else:
            print(f"{sentiment_column} or {rating_column} column not found.")

    def plot_popular_brands_by_category(self):
        """Plots the most popular brands by category with different colors per category."""
        if 'vote' in self.df.columns and 'rating' in self.df.columns and 'brand' in self.df.columns:
            # Group by category and brand, then aggregate by sum of votes, mean of rating, and count of items
            brand_stats = self.df.groupby(['category', 'brand']).agg({
                'vote': 'sum',
                'rating': 'mean',
                'itemName': 'count'
            }).reset_index()

            # Sort within each category and get the top 2 brands per category
            top_brands = brand_stats.sort_values(['category', 'vote'], ascending=[True, False]).groupby(
                'category').head(2)

            # Plotting
            plt.figure(figsize=(18, 12))  # Increased size for better readability
            sns.barplot(
                x='vote', y='brand', hue='category', data=top_brands, dodge=False,
                palette='Set2'
            )

            # Annotate the bars with the average rating and number of items sold on a single line above the bars
            # for index, row in top_brands.iterrows():
            #     plt.text(
            #         row['vote'] + 5,  # Adjust positioning to the right of the bar
            #         index,
            #         f'Rating: {row["rating"]:.2f}, Items: {row["itemName"]}',
            #         va='center', ha='left', fontsize=10, color='black', fontweight='bold'
            #     )

            plt.title('Top 10 Popular Brands by Category', fontsize=16)
            plt.xlabel('Number of Votes', fontsize=14)
            plt.ylabel('Brand', fontsize=14)
            plt.xticks(rotation=45, fontsize=12)  # Rotate x-axis labels if needed
            plt.yticks(fontsize=12)
            plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
        else:
            print("'vote', 'rating', or 'brand' column not found in the dataset.")

    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.pyplot as plt
    import seaborn as sns

    def plot_most_voted_items_per_category(self):
        """Plots the most voted items per category."""
        if 'vote' in self.df.columns:
            most_voted_items = self.df.groupby(['category', 'itemName'])['vote'].sum().reset_index()
            most_voted = most_voted_items.sort_values(by='vote', ascending=False).groupby(
                'category').first().reset_index()

            plt.figure(figsize=(14, 8))
            ax = sns.barplot(x='vote', y='itemName', hue='category', data=most_voted, dodge=False)

            # Remove the default y-axis labels
            ax.set_yticklabels([])

            # Adding text labels aligned inside the bars, close to the left edge of the graph
            for index, row in most_voted.iterrows():
                ax.text(
                    0.2,  # Slightly offset from the start of the graph area
                    index,  # Align text with the corresponding bar
                    row['itemName'],
                    color='black',
                    va='center',
                    ha='left'
                )

            plt.title('Most Voted Items per Category')
            plt.xlabel('Number of Votes')
            plt.ylabel('Item Name')
            plt.legend(title='Category')
            plt.show()
        else:
            print("'vote' column not found in the dataset.")

    def plot_most_expensive_categories_with_votes(self):
        """Plots the most expensive categories with the number of votes."""
        if 'price' in self.df.columns and 'vote' in self.df.columns:
            # Clean the price column: remove non-numeric values and convert to float
            self.df['price'] = pd.to_numeric(self.df['price'].replace('[\$,]', '', regex=True), errors='coerce')

            # Drop rows where price is NaN after conversion
            self.df.dropna(subset=['price'], inplace=True)

            most_expensive_categories = self.df.groupby('category').agg({
                'price': 'max',
                'vote': 'sum'
            }).reset_index()

            plt.figure(figsize=(14, 8))
            sns.scatterplot(x='price', y='vote', hue='category', size='price', data=most_expensive_categories,
                            sizes=(50, 200))
            plt.title('Most Expensive Categories with Number of Votes')
            plt.xlabel('Max Price')
            plt.ylabel('Total Votes')
            plt.legend(title='Category')
            plt.show()
        else:
            print("'price' or 'vote' column not found in the dataset.")

    def plot_most_popular_users_per_category(self):
        """Plots the most popular users per category based on votes."""
        if 'vote' in self.df.columns and 'userName' in self.df.columns:
            # Group by category and userName, then sum the votes
            most_popular_users = self.df.groupby(['category', 'userName'])['vote'].sum().reset_index()

            # Find the user with the maximum votes for each category
            most_popular = most_popular_users.sort_values(by='vote', ascending=False).groupby(
                'category').first().reset_index()

            # Plotting
            plt.figure(figsize=(14, 8))
            sns.barplot(x='vote', y='userName', hue='category', data=most_popular, dodge=False)
            plt.title('Most Popular Users per Category Based on Votes')
            plt.xlabel('Number of Votes')
            plt.ylabel('User Name')
            plt.legend(title='Category')
            plt.show()
        else:
            print("'vote' or 'userName' column not found in the dataset.")

    def plot_price_deviation_anomalies(self):
        """Detects and plots anomalies in price deviation for each category separately."""
        if 'price' in self.df.columns and 'category' in self.df.columns:
            self.df['price'] = pd.to_numeric(self.df['price'].replace('[\$,]', '', regex=True), errors='coerce')
            self.df.dropna(subset=['price'], inplace=True)

            # Detect price deviations per category
            self.df['price_anomaly'] = self.df.groupby('category')['price'].transform(
                lambda x: np.abs(x - x.mean()) > 2 * x.std()
            )

            categories = self.df['category'].unique()

            # Loop through each category and create a separate plot
            for category in categories:
                category_data = self.df[self.df['category'] == category]
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=category_data, x='price', y='category', hue='price_anomaly',
                                palette={True: 'red', False: 'blue'})
                plt.title(f'Price Deviation Anomalies for {category}')
                plt.xlabel('Price')
                plt.ylabel('Category')
                plt.legend(title='Anomaly Detected', loc='upper right')
                plt.show()

        else:
            print("'price' or 'category' column not found in the dataset.")

    def plot_irregular_feedback_anomalies(self):
        """Detects and plots anomalies in rating per category with categories on the Y-axis and the legend outside the graph."""
        if 'rating' in self.df.columns and 'category' in self.df.columns:
            # Detect irregular feedback per category
            self.df['rating_anomaly'] = self.df.groupby('category')['rating'].transform(
                lambda x: np.abs(x - x.mean()) > 2 * x.std()
            )

            plt.figure(figsize=(14, 8))
            sns.scatterplot(data=self.df, x='rating', y='category', hue='category', style='rating_anomaly',
                            palette='Set1', markers={True: 'X', False: 'o'})

            plt.title('Irregular Rating Anomalies per Category')
            plt.xlabel('Rating Score')
            plt.ylabel('Category')

            # Position the legend outside the plot
            plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()  # Adjust the layout to make room for the legend

            plt.show()
        else:
            print("'rating' or 'category' column not found in the dataset.")

    def run_all_visualizations(self):
        self.preprocess_data()
       # self.plot_sentiment_by_category()
        #self.plot_helpful_votes_distribution()
       # self.plot_rating_discrepancies()

       # self.plot_popular_brands_by_category()
        #self.plot_most_voted_items_per_category()
        #self.plot_most_expensive_categories_with_votes()
        #self.plot_most_popular_users_per_category()
        self.plot_irregular_feedback_anomalies()
        #self.plot_price_deviation_anomalies()


def main():
    file_path = './data-sets/amazon_reviews.csv'
    visualizer = DataVisualizer(file_path)
    visualizer.run_all_visualizations()


if __name__ == "__main__":
    main()
