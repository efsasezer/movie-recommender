# Advanced Movie Recommendation System 🎬

A sophisticated movie recommendation system built with Streamlit, offering personalized movie suggestions using collaborative filtering, popularity-based recommendations, and hybrid approaches.

Live Demo: [ [Movie Recommender]](https://vmfinal-seqpv2kn9xoyaqrcxsictd.streamlit.app/)

## Features 🌟

- **Movie-Based Recommendations**: Get personalized movie suggestions based on similar movies using collaborative filtering
- **Popular Movies**: Discover top-rated movies filtered by minimum rating count
- **Hybrid Recommendations**: Combined approach using both similarity and popularity metrics
- **Data Analysis**: Interactive visualizations of movie ratings, genre distributions, and other metrics

## Technologies Used 🛠️

- Python 3.x
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly Express

## Installation 🔧

1. Clone the repository:
```bash
git clone [Your Repository URL]
cd movie-recommender
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the required datasets (`movies.csv` and `ratings.csv`) and place them in the project root directory.

4. Run the application:
```bash
streamlit run app.py
```

## Dataset Requirements 📊

The system requires two CSV files:
- `movies.csv`: Contains movie information (movieId, title, genres)
- `ratings.csv`: Contains user ratings (userId, movieId, rating)

## Features in Detail 📝

### 1. Movie-Based Recommendations
- Uses collaborative filtering with cosine similarity
- Provides similarity scores for recommended movies
- Interactive visualization of similarity metrics

### 2. Popular Movies
- Configurable minimum rating threshold
- Sorted by average rating
- Detailed statistics and visualizations

### 3. Hybrid Recommendations
- Combines similarity-based and popularity-based approaches
- Balanced recommendations for better user experience

### 4. Data Analysis
- Rating distribution visualization
- Genre analysis
- User activity metrics
- Interactive charts and graphs

## Project Structure 📁

```
movie-recommender/
├── app.py
├── movies.csv
├── ratings.csv
├── requirements.txt
└── README.md
```

## Usage Example 💡

1. Navigate to the web application
2. Select a page from the sidebar navigation
3. For movie-based recommendations:
   - Select a movie from the dropdown
   - Click "Show Recommendations"
   - View similar movies with similarity scores
4. For popular movies:
   - Adjust the minimum ratings slider
   - View top-rated movies and statistics

## Contributing 🤝

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

