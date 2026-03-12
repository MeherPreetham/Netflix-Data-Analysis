# Netflix Movies and TV Shows Data Analysis

> Comprehensive exploratory data analysis of Netflix's content library with genre classification, geographical distribution, temporal trends, and an intelligent content recommendation system.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green.svg)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)

## Project Overview

This project analyses Netflix's movie and TV show catalogue using Python data science libraries to uncover insights about:

- **Content Distribution**: Genre and age rating classifications
- **Geographic Trends**: Country-wise content production patterns  
- **Temporal Analysis**: Duration and genre trends over decades
- **Smart Recommendations**: TF-IDF based content recommendation engine

**Dataset**: 8,807 Netflix titles from [Kaggle](https://www.kaggle.com/) (movies and TV shows up to 2021)

## Key Features

### 1. Genre & Age Rating Analysis
- Top genre distribution visualisation
- Age rating frequency analysis
- Genre-age rating correlation heatmaps

### 2. Geographical Distribution
- Content production by country (top producers)
- Genre preferences across different countries
- Country-specific age rating patterns

### 3. Duration & Temporal Trends
- Average content duration evolution over decades
- Genre popularity trends across time periods
- Age restriction patterns over time
- Netflix acquisition delay analysis

### 4. Intelligent Recommendation System
- TF-IDF vectorisation of content features
- Cosine similarity-based recommendations
- Combines description, genre, cast, and director metadata

## Quick Start

### Prerequisites

```bash
pip install pandas matplotlib seaborn scikit-learn
```

### Usage

```python
from netflix_analyzer import NetflixAnalyzer

# Initialize analyzer
analyzer = NetflixAnalyzer('netflix_titles.csv')

# View dataset information
analyzer.get_dataset_info()

# Genre analysis
analyzer.plot_genre_distribution(top_n=10)
analyzer.plot_genre_age_heatmap()

# Geographical analysis
analyzer.plot_content_by_country(top_n=10)
analyzer.plot_country_age_rating_heatmap()

# Temporal trends
analyzer.analyze_duration_trends()
analyzer.plot_genre_trends_by_decade()

# Get recommendations
analyzer.build_recommendation_system()
recommendations = analyzer.get_recommendations('Stranger Things', top_n=10)
print(recommendations)
```

## Repository Structure

```
netflix-analysis/
├── netflix_analyzer.py      # Main analysis class with all methods
├── requirements.txt          # Python dependencies
├── netflix_titles.csv        # Dataset (download from Kaggle)
├── README.md                # This file
└── examples/                # Example notebooks (optional)
```

## Sample Insights

**Top Findings:**
- The United States produces the most Netflix content (over 2,500 titles)
- Dramas and Comedies dominate genre distribution
- Average movie duration has decreased over recent decades
- Content acquisition delay has reduced significantly post-2015

## Technologies Used

- **Python 3.8+** - Core programming language
- **Pandas** - Data manipulation and analysis
- **Matplotlib & Seaborn** - Data visualization
- **scikit-learn** - TF-IDF vectorisation and similarity computation

## Analysis Categories

| Analysis Type | Methods | Visualizations |
|--------------|---------|----------------|
| **Genre Analysis** | `plot_genre_distribution()`, `plot_age_rating_distribution()`, `plot_genre_age_heatmap()` | Bar charts, Heatmaps |
| **Geographic** | `plot_content_by_country()`, `plot_country_genre_scatter()` | Stacked bar charts, Scatter plots |
| **Temporal** | `analyze_duration_trends()`, `plot_genre_trends_by_decade()` | Line plots, Trend analysis |
| **Recommendations** | `build_recommendation_system()`, `get_recommendations()` | Similarity scores, Bar charts |

## Learning Outcomes

- **Data Cleaning**: Handling missing values, string normalisation, duplicate detection
- **Exploratory Data Analysis**: Statistical summaries, distribution analysis, correlation studies
- **Visualisation**: Creating publication-quality charts with Matplotlib/Seaborn
- **Machine Learning**: Implementing TF-IDF and cosine similarity for recommendations
- **Python OOP**: Building scalable analysis pipelines with class-based architecture

## Example Use Cases

**Content Discovery:**
```python
# Find similar shows to "Breaking Bad"
analyzer.get_recommendations('Breaking Bad', top_n=5)
```

**Market Analysis:**
```python
# Analyse content strategy by country
analyzer.plot_content_by_country(top_n=15)
```

**Trend Forecasting:**
```python
# Understand genre evolution
analyzer.plot_genre_trends_by_decade(top_n=5)
```

## Contributing

This project was developed as part of academic coursework. Suggestions and improvements are welcome!

## License

This project is open source and available for educational purposes.

## Author

**Meher Preetham Kommera**  
Master's in Data Science / Cloud Computing  
[LinkedIn](https://www.linkedin.com/in/meher-preetham-kommera-23184023a/) • [GitHub](https://github.com/MeherPreetham)

---

**Academic Project**: COMP5721M - Programming for Data Science  
**Institution**: [University of Leeds]  
**Year**: 2024

## Acknowledgments

- Dataset provided by Shivam Bansal on Kaggle
- Course instructors and teaching staff
- Netflix for the rich content metadata
