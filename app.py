import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import os
import zipfile

class MovieRecommender:
    def __init__(self):
        self.load_data()
        self.prepare_features()
        
    def load_data(self):
        try:
            # Veri setlerini yükle
            self.movies = pd.read_csv('./movie.csv')
            self.ratings = pd.read_csv('./rating.csv')
            # Gereksiz sütunları kaldır
            if "timestamp" in self.ratings.columns:
                self.ratings = self.ratings.drop("timestamp", axis=1)

            # Veri setini küçült (performans için)
            self.ratings = self.ratings.head(100000)
            self.movies = self.movies[
                self.movies["movieId"].isin(self.ratings["movieId"].unique())
            ]
        except Exception as e:
            st.error(f"Veri yükleme hatası: {str(e)}")
            raise


    def prepare_features(self):
        try:
            # Popülerlik istatistikleri
            self.movie_stats = self.ratings.groupby('movieId').agg({
                'rating': ['count', 'mean']
            }).reset_index()
            self.movie_stats.columns = ['movieId', 'rating_count', 'rating_mean']
            
            # Film-kullanıcı matrisi
            ratings_matrix = self.ratings.pivot(
                index='movieId',
                columns='userId',
                values='rating'
            ).fillna(0)
            
            self.movie_features = ratings_matrix.values
            self.movie_ids = ratings_matrix.index
            
            # Movie ID mapping
            self.movie_id_to_idx = {
                movie_id: idx for idx, movie_id in enumerate(self.movie_ids)
            }
            
            # Film özellikleri matrisi
            self.movie_features_matrix = ratings_matrix
            
        except Exception as e:
            st.error(f"Özellik hazırlama hatası: {str(e)}")
            raise
    
    def get_recommendations(self, movie_id, n=5):
        try:
            if movie_id not in self.movie_id_to_idx:
                return pd.DataFrame()
                
            idx = self.movie_id_to_idx[movie_id]
            movie_vec = self.movie_features[idx]
            
            similarities = cosine_similarity([movie_vec], self.movie_features)[0]
            similar_indices = similarities.argsort()[::-1][1:n+1]
            similar_movies = self.movie_ids[similar_indices]
            similar_scores = similarities[similar_indices]
            
            recommendations = []
            for movie_id, similarity in zip(similar_movies, similar_scores):
                movie_info = self.movies[self.movies['movieId'] == movie_id].iloc[0]
                recommendations.append({
                    'Title': movie_info['title'],
                    'Genre': movie_info['genres'],
                    'Similarity': similarity
                })
                
            return pd.DataFrame(recommendations)
            
        except Exception as e:
            st.error(f"Öneri hesaplama hatası: {str(e)}")
            return pd.DataFrame()
    
    def get_popular_movies(self, n=10, min_ratings=50):
        try:
            popular = self.movie_stats[self.movie_stats['rating_count'] >= min_ratings]
            popular = popular.sort_values('rating_mean', ascending=False)
            
            return popular.merge(self.movies, on='movieId')[
                ['title', 'genres', 'rating_mean', 'rating_count']
            ].head(n)
            
        except Exception as e:
            st.error(f"Popüler film hesaplama hatası: {str(e)}")
            return pd.DataFrame()
    
    def get_hybrid_recommendations(self, movie_id, n=5):
        similar = self.get_recommendations(movie_id, n=n)
        popular = self.get_popular_movies(n=n)
        
        hybrid = pd.concat([
            similar.head(n//2),
            popular.head(n-n//2)
        ]).reset_index(drop=True)
        
        return hybrid

def main():
    st.set_page_config(page_title="Film Öneri Sistemi", layout="wide")
    
    # Sidebar stil
    st.markdown("""
        <style>
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Ana başlık
    st.title("🎬 Film Öneri Sistemi")
    
    try:
        recommender = MovieRecommender()
        
        # Navbar
        st.sidebar.title("📚 Navigasyon")
        pages = {
            "🏠 Ana Sayfa": "home",
            "🎯 Film Bazlı Öneriler": "movie_based",
            "🌟 Popüler Filmler": "popular",
            "🔄 Karma Öneriler": "hybrid",
            "📊 Veri Analizi": "analysis",
            "ℹ️ Hakkında": "about"
        }
        
        page = st.sidebar.radio("Sayfa Seçin:", list(pages.keys()))
        
        # Ana Sayfa
        if pages[page] == "home":
            st.header("Film Öneri Sistemine Hoş Geldiniz!")
            st.write("""
            Bu uygulama, film önerileri için çeşitli yöntemler sunar:
            - Film bazlı öneriler
            - Popülerlik bazlı öneriler
            - Karma öneriler
            - Detaylı veri analizi
            """)
            
            # Temel istatistikler
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Toplam Film", len(recommender.movies))
            with col2:
                st.metric("Toplam Kullanıcı", recommender.ratings['userId'].nunique())
            with col3:
                st.metric("Toplam Değerlendirme", len(recommender.ratings))
        
        # Film Bazlı Öneriler
        elif pages[page] == "movie_based":
            st.header("🎯 Film Bazlı Öneriler")
            
            selected_movie = st.selectbox(
                "Film seçin:",
                recommender.movies['title'].tolist()
            )
            
            if st.button("Önerileri Göster"):
                with st.spinner("Öneriler hesaplanıyor..."):
                    movie_id = recommender.movies[
                        recommender.movies['title'] == selected_movie
                    ]['movieId'].iloc[0]
                    
                    recommendations = recommender.get_recommendations(movie_id)
                    
                    if not recommendations.empty:
                        # Benzerlik grafiği
                        fig = px.bar(
                            recommendations,
                            x='Title',
                            y='Similarity',
                            title=f"{selected_movie} Filmine Benzer Filmler",
                            color='Similarity'
                        )
                        st.plotly_chart(fig)
                        
                        # Detaylı tablo
                        st.dataframe(recommendations)
                        
        # Popüler Filmler
        elif pages[page] == "popular":
            st.header("🌟 Popüler Filmler")
            
            min_ratings = st.slider(
                "Minimum değerlendirme sayısı:",
                min_value=10,
                max_value=100,
                value=50
            )
            
            with st.spinner("Popüler filmler yükleniyor..."):
                popular_movies = recommender.get_popular_movies(min_ratings=min_ratings)
                
                if not popular_movies.empty:
                    # Popülerlik grafiği
                    fig = px.bar(
                        popular_movies,
                        x='title',
                        y='rating_mean',
                        color='rating_count',
                        title="En Yüksek Puanlı Filmler"
                    )
                    st.plotly_chart(fig)
                    
                    # Detaylı tablo
                    st.dataframe(popular_movies)
        
        # Karma Öneriler
        elif pages[page] == "hybrid":
            st.header("🔄 Karma Öneriler")
            
            selected_movie = st.selectbox(
                "Film seçin:",
                recommender.movies['title'].tolist()
            )
            
            if st.button("Karma Önerileri Göster"):
                with st.spinner("Öneriler hazırlanıyor..."):
                    movie_id = recommender.movies[
                        recommender.movies['title'] == selected_movie
                    ]['movieId'].iloc[0]
                    
                    hybrid = recommender.get_hybrid_recommendations(movie_id)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Benzer Filmler")
                        st.dataframe(hybrid.iloc[:len(hybrid)//2])
                        
                    with col2:
                        st.subheader("Popüler Filmler")
                        st.dataframe(hybrid.iloc[len(hybrid)//2:])
        
        # Veri Analizi
        elif pages[page] == "analysis":
            st.header("📊 Veri Analizi")
            
            # Genel istatistikler
            st.subheader("Genel İstatistikler")
            col1, col2 = st.columns(2)
            
            with col1:
                # Puan dağılımı
                fig = px.histogram(
                    recommender.ratings,
                    x='rating',
                    title="Puan Dağılımı"
                )
                st.plotly_chart(fig)
                
            with col2:
                # Film başına değerlendirme sayısı
                fig = px.histogram(
                    recommender.movie_stats,
                    x='rating_count',
                    title="Film Başına Değerlendirme Sayısı"
                )
                st.plotly_chart(fig)
            
            # Tür analizi
            st.subheader("Film Türleri Analizi")
            genres = recommender.movies['genres'].str.split('|', expand=True).stack()
            genre_counts = genres.value_counts()
            
            fig = px.pie(
                values=genre_counts.values,
                names=genre_counts.index,
                title="Film Türleri Dağılımı"
            )
            st.plotly_chart(fig)
        
        # Hakkında
        else:
            st.header("ℹ️ Hakkında")
            st.write("""
            Bu film öneri sistemi, çeşitli algoritmalar kullanarak kişiselleştirilmiş
            film önerileri sunar. Sistem şunları içerir:
            
            - İçerik tabanlı öneriler
            - İşbirlikçi filtreleme
            - Popülerlik bazlı öneriler
            - Karma öneriler
            
            Veriler düzenli olarak güncellenmektedir.
            """)
            
    except Exception as e:
        st.error(f"Uygulama hatası: {str(e)}")
        st.error("Lütfen veri setlerinin doğru yüklendiğinden emin olun.")

if __name__ == "__main__":
    main()
