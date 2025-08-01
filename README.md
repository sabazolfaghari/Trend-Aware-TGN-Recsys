## Trend-Aware-TGN-Recsys
Trend-Aware-TGN-Recsys is a **trend-aware movie recommender system** built on the **Temporal Graph Network (TGN)** architecture.  
Unlike static recommenders, this model integrates **temporal dynamics of IMDb popularity trends, ratings, and movie metadata (genres, directors, actors, plot embeddings, and ...)** with user–item interactions to predict the ratings a user would give to a movie.
This approach captures both **short-term popularity shifts** and **long-term user preferences**, making recommendations more adaptive and realistic.

## Dataset
This project uses the **[MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/)**:

- **100,000 ratings** (1–5 stars)  
- **943 users** and **1,682 movies**  
- Includes timestamps and basic movie metadata (titles, release years, genres)  

Due to licensing, the dataset is **not included** in this repository.  
Please download it from the [official MovieLens website](https://grouplens.org/datasets/movielens/100k/) and use the files u.data and u.item.

## Installation
This project is implemented with Python 3.11.0.

Install dependencies with pip install. Example:
pip install torch-geometric

## Usage
```bash
python Trend-Aware TGN.py
