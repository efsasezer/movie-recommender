import os
import gdown
import pandas as pd
import streamlit as st

class MovieRecommender:
    def __init__(self):
        self.movie_url = 'https://drive.google.com/uc?id=1lvWbK8UBRD6TOar3etP0IU6gxv52YR72'
        self.rating_url = 'https://drive.google.com/uc?id=1MLupnNHLXXLr6N-oXHzw02MvvXvrTCzO'
        self.movie_file = 'movie.csv'
        self.rating_file = 'rating.csv'
        self.movies = None
        self.ratings = None

    def download_data(self):
        """Download movie and rating datasets from Google Drive."""
        if not os.path.exists(self.movie_file):
            st.info("Downloading movie dataset...")
            gdown.download(self.movie_url, self.movie_file, quiet=False)

        if not os.path.exists(self.rating_file):
            st.info("Downloading rating dataset...")
            gdown.download(self.rating_url, self.rating_file, quiet=False)

        if os.path.exists(self.movie_file) and os.path.exists(self.rating_file):
            st.success("Datasets downloaded successfully!")
        else:
            st.error("Failed to download datasets. Please check the URLs and permissions.")

    def load_data(self):
        """Load the datasets into Pandas DataFrames."""
        try:
            self.movies = pd.read_csv(self.movie_file)
            self.ratings = pd.read_csv(self.rating_file)
            st.success("Datasets loaded successfully!")
        except Exception as e:
            st.error(f"Error loading datasets: {e}")

    def recommend_movies(self, movie_title, top_n=5):
        """Recommend movies based on a given movie title."""
        if self.movies is None or self.ratings is None:
            st.error("Data not loaded. Please ensure datasets are available.")
            return []

        # Merge movie and rating data
        data = pd.merge(self.ratings, self.movies, on='movieId')

        # Calculate mean rating and rating count for each movie
        movie_stats = data.groupby('title').agg({'rating': ['mean', 'count']})
        movie_stats.columns = ['mean_rating', 'rating_count']
        movie_stats = movie_stats.reset_index()

        # Find the target movie
        target_movie = movie_stats[movie_stats['title'].str.contains(movie_title, case=False, na=False)]
        if target_movie.empty:
            st.warning("Movie not found in the dataset.")
            return []

        target_movie_name = target_movie.iloc[0]['title']
        st.info(f"Target movie: {target_movie_name}")

        # Calculate similarity based on rating
        data_pivot = data.pivot_table(index='userId', columns='title', values='rating')
        target_ratings = data_pivot[target_movie_name]
        similarity = data_pivot.corrwith(target_ratings)

        # Create a similarity DataFrame
        similarity_df = similarity.dropna().reset_index()
        similarity_df.columns = ['title', 'similarity']

        # Merge with movie stats for better recommendations
        recommendations = pd.merge(similarity_df, movie_stats, on='title')
        recommendations = recommendations[recommendations['title'] != target_movie_name]
        recommendations = recommendations.sort_values(by=['similarity', 'mean_rating'], ascending=False)

        return recommendations.head(top_n)

# Streamlit UI
def main():
    st.title("Movie Recommendation System")

    recommender = MovieRecommender()

    # Download and load data
    if st.button("Download and Load Data"):
        recommender.download_data()
        recommender.load_data()

    # Movie recommendation
    movie_title = st.text_input("Enter a movie title you like:")
    top_n = st.number_input("Number of recommendations:", min_value=1, max_value=20, value=5)

    if st.button("Recommend Movies"):
        if movie_title:
            recommendations = recommender.recommend_movies(movie_title, top_n)
            if recommendations:
                st.subheader("Recommended Movies")
                st.write(recommendations[['title', 'mean_rating', 'rating_count']])
        else:
            st.warning("Please enter a movie title.")

if __name__ == "__main__":
    main()
