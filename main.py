"""
Netflix Movies and TV Shows Data Analysis
==========================================

A comprehensive data analysis project exploring Netflix's content library,
including genre classification, geographical distribution, duration trends,
and a content recommendation system.

Author: [Your Name]
Course: COMP5721M - Programming for Data Science
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class NetflixAnalyzer:
    """Main class for Netflix data analysis"""
    
    def __init__(self, filepath='netflix_titles.csv'):
        """Initialize analyzer with dataset"""
        self.df = pd.read_csv(filepath)
        self._clean_data()
    
    def _clean_data(self):
        """Clean and preprocess the dataset"""
        self.df['listed_in'] = self.df['listed_in'].str.strip().str.lower()
        self.df['rating'] = self.df['rating'].str.strip()
    
    def get_dataset_info(self):
        """Display dataset information and statistics"""
        print("\n=== Dataset Information ===")
        print(self.df.info())
        print("\n=== Missing Values ===")
        print(self.df.isnull().sum())
        print("\n=== First 5 Rows ===")
        print(self.df.head())
        return self.df
    
    # ================== Genre Analysis ==================
    
    def plot_genre_distribution(self, top_n=10):
        """Plot distribution of top genres"""
        all_genres = self.df['listed_in'].str.split(',').explode().str.strip()
        genre_counts = all_genres.value_counts()
        top_genres = genre_counts.head(top_n)
        
        plt.figure(figsize=(10, 6))
        plt.barh(top_genres.index, top_genres.values, color='skyblue')
        plt.title(f'Top {top_n} Genre Distribution', fontsize=14)
        plt.xlabel('Count', fontsize=12)
        plt.ylabel('Genre', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    def plot_age_rating_distribution(self, top_n=10):
        """Plot distribution of age ratings"""
        age_rating_counts = self.df['rating'].value_counts()
        top_age_ratings = age_rating_counts.head(top_n)
        
        plt.figure(figsize=(10, 6))
        plt.barh(top_age_ratings.index, top_age_ratings.values, color='lightgreen')
        plt.title(f'Top {top_n} Age Rating Distribution', fontsize=14)
        plt.xlabel('Count', fontsize=12)
        plt.ylabel('Age Rating', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    def plot_genre_age_heatmap(self, top_n=10):
        """Create heatmap showing genre vs age rating correlation"""
        excluded_ratings = ["66 min", "74 min", "84 min"]
        filtered_data = self.df[~self.df['rating'].isin(excluded_ratings)]
        
        exploded_data = filtered_data.copy()
        exploded_data = exploded_data.assign(
            listed_in=exploded_data['listed_in'].str.split(',')
        ).explode('listed_in')
        exploded_data['listed_in'] = exploded_data['listed_in'].str.strip()
        
        pivot_data = exploded_data.groupby(['listed_in', 'rating']).size().unstack(fill_value=0)
        top_genres = pivot_data.sum(axis=1).nlargest(top_n).index
        pivot_data = pivot_data.loc[top_genres]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot_data,
            cmap='coolwarm',
            annot=True,
            fmt='.0f',
            cbar_kws={'label': 'Count'},
            linewidths=0.5,
            annot_kws={'size': 10}
        )
        plt.title('Genre vs Age Rating Heatmap', fontsize=16)
        plt.xlabel('Age Rating', fontsize=12)
        plt.ylabel('Genre', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    # ================== Geographical Analysis ==================
    
    def plot_content_by_country(self, top_n=10):
        """Plot content distribution by country"""
        exploded_data = self.df.copy()
        exploded_data = exploded_data.assign(
            listed_in=exploded_data['listed_in'].str.split(',')
        ).explode('listed_in')
        exploded_data['listed_in'] = exploded_data['listed_in'].str.strip()
        
        top_countries = exploded_data['country'].value_counts().nlargest(top_n)
        top_countries_list = top_countries.index
        
        content_type_by_country = (
            exploded_data[exploded_data['country'].isin(top_countries_list)]
            .groupby(['country', 'type'])
            .size()
            .unstack(fill_value=0)
        )
        
        content_type_by_country['Total'] = content_type_by_country.sum(axis=1)
        content_type_by_country = content_type_by_country.sort_values(by='Total', ascending=False)
        content_type_by_country = content_type_by_country.drop(columns=['Total'])
        
        content_type_by_country.plot(
            kind='bar',
            stacked=True,
            figsize=(14, 7),
            color=['#37e418', '#3c82e7'],
            edgecolor='black'
        )
        plt.title(f'Content Distribution by Country (Top {top_n})', fontsize=16)
        plt.xlabel('Country', fontsize=12)
        plt.ylabel('Number of Titles', fontsize=12)
        plt.legend(title='Content Type', fontsize=10)
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
    
    def plot_country_genre_scatter(self, top_n=100):
        """Scatter plot of countries vs genres"""
        exploded_data = self.df.copy()
        exploded_data = exploded_data.assign(
            country=self.df['country'].str.split(',')
        ).explode('country')
        exploded_data['country'] = exploded_data['country'].str.strip()
        
        exploded_data = exploded_data.assign(
            listed_in=exploded_data['listed_in'].str.split(',')
        ).explode('listed_in')
        exploded_data['listed_in'] = exploded_data['listed_in'].str.strip()
        
        pivot_table = exploded_data.groupby(['country', 'listed_in']).size().reset_index(name='count')
        top_combinations = pivot_table.nlargest(top_n, 'count')
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            top_combinations['country'],
            top_combinations['listed_in'],
            s=top_combinations['count'] * 2,
            alpha=0.6,
            c=top_combinations['count'],
            cmap='viridis'
        )
        plt.colorbar(scatter, label='Number of Shows')
        plt.title(f'Top {top_n} Country vs Genres', fontsize=16)
        plt.xlabel('Country', fontsize=12)
        plt.ylabel('Genres', fontsize=12)
        plt.xticks(rotation=60)
        plt.tight_layout()
        plt.show()
    
    def plot_country_age_rating_heatmap(self, top_n=10):
        """Heatmap of country vs age rating distribution"""
        top_countries = self.df['country'].value_counts().nlargest(top_n).index
        filtered_data = self.df[self.df['country'].isin(top_countries)]
        
        country_age_rating_pivot = (
            filtered_data.groupby(['country', 'rating'])
            .size()
            .unstack(fill_value=0)
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            country_age_rating_pivot,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            linewidths=0.5
        )
        plt.title(f'Country vs Age Rating Distribution (Top {top_n})', fontsize=16)
        plt.xlabel('Age Rating', fontsize=12)
        plt.ylabel('Country', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    # ================== Duration Analysis ==================
    
    def analyze_duration_trends(self):
        """Analyze content duration trends over time"""
        df_copy = self.df.copy()
        df_copy['duration_minutes'] = df_copy['duration'].str.extract(r'(\d+)').astype(float)
        df_copy = df_copy.dropna(subset=['duration_minutes', 'release_year'])
        df_copy['decade'] = (df_copy['release_year'] // 10) * 10
        
        # Calculate average duration per decade
        duration_trend = df_copy.groupby('decade')['duration_minutes'].mean().reset_index()
        
        plt.figure(figsize=(12, 6))
        plt.plot(duration_trend['decade'], duration_trend['duration_minutes'], 
                marker='o', linestyle='-', linewidth=2, markersize=8, color='teal')
        plt.title('Average Content Duration Over Decades', fontsize=16)
        plt.xlabel('Decade', fontsize=12)
        plt.ylabel('Average Duration (minutes)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
        
        return duration_trend
    
    def plot_genre_trends_by_decade(self, top_n=5):
        """Analyze genre popularity trends across decades"""
        df_copy = self.df.copy()
        df_copy = df_copy.dropna(subset=['release_year'])
        df_copy['decade'] = (df_copy['release_year'] // 10) * 10
        
        exploded_data = df_copy.assign(
            listed_in=df_copy['listed_in'].str.split(',')
        ).explode('listed_in')
        exploded_data['listed_in'] = exploded_data['listed_in'].str.strip()
        
        top_genres = exploded_data['listed_in'].value_counts().head(top_n).index
        filtered_data = exploded_data[exploded_data['listed_in'].isin(top_genres)]
        
        genre_decade_counts = (
            filtered_data.groupby(['decade', 'listed_in'])
            .size()
            .unstack(fill_value=0)
        )
        
        genre_decade_counts.plot(kind='line', figsize=(14, 7), marker='o', linewidth=2)
        plt.title(f'Genre Trends Over Decades (Top {top_n})', fontsize=16)
        plt.xlabel('Decade', fontsize=12)
        plt.ylabel('Number of Titles', fontsize=12)
        plt.legend(title='Genre', fontsize=10, loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
    
    def plot_age_restriction_trends(self):
        """Analyze age restriction trends over decades"""
        df_copy = self.df.copy()
        df_copy = df_copy.dropna(subset=['release_year'])
        df_copy['decade'] = (df_copy['release_year'] // 10) * 10
        
        age_rating_mapping = {
            'G': 'General Audiences',
            'PG': 'Parental Guidance',
            'PG-13': 'Parents Strongly Cautioned',
            'R': 'Restricted',
            'TV-Y': 'All Children',
            'TV-Y7': 'Directed to Older Children',
            'TV-G': 'General Audiences',
            'TV-PG': 'Parental Guidance Suggested',
            'TV-14': 'Parents Strongly Cautioned',
            'TV-MA': 'Mature Audiences'
        }
        
        df_copy['age_category'] = df_copy['rating'].map(age_rating_mapping)
        df_copy = df_copy.dropna(subset=['age_category'])
        
        age_decade_counts = (
            df_copy.groupby(['decade', 'age_category'])
            .size()
            .unstack(fill_value=0)
        )
        
        age_decade_counts.plot(kind='line', figsize=(14, 7), marker='o', linewidth=2)
        plt.title('Age Restriction Trends Over Decades', fontsize=16)
        plt.xlabel('Decade', fontsize=12)
        plt.ylabel('Number of Titles', fontsize=12)
        plt.legend(title='Age Category', fontsize=10, loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
    
    def analyze_acquisition_delay(self):
        """Analyze the delay between content release and Netflix acquisition"""
        df_copy = self.df.copy()
        df_copy = df_copy.dropna(subset=['release_year', 'date_added'])
        df_copy['date_added'] = pd.to_datetime(df_copy['date_added'])
        df_copy['year_added'] = df_copy['date_added'].dt.year
        df_copy['acquisition_delay'] = df_copy['year_added'] - df_copy['release_year']
        df_copy = df_copy[df_copy['acquisition_delay'] >= 0]
        
        avg_delay = df_copy.groupby('year_added')['acquisition_delay'].mean().reset_index()
        
        plt.figure(figsize=(14, 6))
        plt.plot(avg_delay['year_added'], avg_delay['acquisition_delay'],
                marker='o', linestyle='-', linewidth=2, markersize=8, color='coral')
        plt.title('Average Acquisition Delay (Years Between Release and Netflix Addition)', fontsize=16)
        plt.xlabel('Year Added to Netflix', fontsize=12)
        plt.ylabel('Average Delay (Years)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
        
        return avg_delay
    
    # ================== Recommendation System ==================
    
    def build_recommendation_system(self):
        """Build TF-IDF based content recommendation system"""
        df_copy = self.df.dropna(subset=['description', 'listed_in', 'cast', 'director'])
        
        # Combine features for recommendation
        df_copy['combined_features'] = (
            df_copy['description'] + ' ' +
            df_copy['listed_in'] + ' ' +
            df_copy['cast'] + ' ' +
            df_copy['director']
        )
        
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(df_copy['combined_features'])
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        self.recommendation_data = {
            'df': df_copy,
            'cosine_sim': cosine_sim,
            'indices': pd.Series(df_copy.index, index=df_copy['title']).drop_duplicates()
        }
        
        print("✓ Recommendation system built successfully!")
        return self.recommendation_data
    
    def get_recommendations(self, title, top_n=10):
        """Get content recommendations based on title"""
        if not hasattr(self, 'recommendation_data'):
            print("Building recommendation system...")
            self.build_recommendation_system()
        
        try:
            idx = self.recommendation_data['indices'][title]
            sim_scores = list(enumerate(self.recommendation_data['cosine_sim'][idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:top_n+1]
            
            movie_indices = [i[0] for i in sim_scores]
            recommendations = self.recommendation_data['df'].iloc[movie_indices][['title', 'type', 'listed_in', 'rating']]
            
            return recommendations
        
        except KeyError:
            print(f"Title '{title}' not found in dataset.")
            print("\nSuggested titles:")
            suggestions = self.recommendation_data['df']['title'].str.contains(
                title.split()[0], case=False, na=False
            )
            print(self.recommendation_data['df'][suggestions]['title'].head(5).tolist())
            return None
    
    def visualize_recommendations(self, title, top_n=10):
        """Visualize recommendation similarities"""
        if not hasattr(self, 'recommendation_data'):
            self.build_recommendation_system()
        
        try:
            idx = self.recommendation_data['indices'][title]
            sim_scores = list(enumerate(self.recommendation_data['cosine_sim'][idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:top_n+1]
            
            titles = [self.recommendation_data['df'].iloc[i[0]]['title'] for i in sim_scores]
            scores = [i[1] for i in sim_scores]
            
            plt.figure(figsize=(12, 6))
            plt.barh(titles, scores, color='mediumpurple')
            plt.xlabel('Similarity Score', fontsize=12)
            plt.ylabel('Recommended Title', fontsize=12)
            plt.title(f'Top {top_n} Recommendations for "{title}"', fontsize=14)
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
        except KeyError:
            print(f"Title '{title}' not found in dataset.")
            return None


# ================== Main Execution ==================

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = NetflixAnalyzer('netflix_titles.csv')
    
    print("Netflix Data Analysis System")
    print("=" * 50)
    
    # Display dataset information
    analyzer.get_dataset_info()
    
    # Example usage - uncomment to run specific analyses:
    
    # Genre Analysis
    # analyzer.plot_genre_distribution()
    # analyzer.plot_age_rating_distribution()
    # analyzer.plot_genre_age_heatmap()
    
    # Geographical Analysis
    # analyzer.plot_content_by_country()
    # analyzer.plot_country_genre_scatter()
    # analyzer.plot_country_age_rating_heatmap()
    
    # Duration Analysis
    # analyzer.analyze_duration_trends()
    # analyzer.plot_genre_trends_by_decade()
    # analyzer.plot_age_restriction_trends()
    # analyzer.analyze_acquisition_delay()
    
    # Recommendation System
    # analyzer.build_recommendation_system()
    # recommendations = analyzer.get_recommendations('Stranger Things', top_n=10)
    # print(recommendations)
    # analyzer.visualize_recommendations('Stranger Things')
