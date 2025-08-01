import os
import re
import json
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from imdb import IMDb
from torch_geometric.data import Data
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import LastNeighborLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch_geometric.nn.models.tgn import TGNMemory, IdentityMessage, LastAggregator

ia = IMDb()  # Initialize globally

class TGNRecommender(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes):
        super().__init__()

        # Corrected IdentityMessage Initialization
        self.msg_passing = IdentityMessage(
            raw_msg_dim=in_channels,
            memory_dim=hidden_channels,  
            time_dim=out_channels
        )

        # Aggregator Module
        self.aggregator = LastAggregator()  

        self.memory = TGNMemory(
            num_nodes=num_nodes,  
            raw_msg_dim=in_channels,  
            memory_dim=hidden_channels,  
            time_dim=out_channels,
            message_module=self.msg_passing,  # Message Passing Module
            aggregator_module=self.aggregator  # Aggregation Module
        )

        # Graph Convolution (Transformer-based)
        self.conv = TransformerConv(hidden_channels, out_channels, heads=4)
        
        # Projection for Saving Embeddings
        self.projection = nn.Linear(out_channels * 2 * 4, out_channels)
        
        # MLP for Rating Prediction
        self.mlp = nn.Sequential(
            nn.Linear(out_channels * 2 * 4, 128),  # Concatenated user-item embedding (x2) and x4 for number of heads
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )
        
    def forward(self, edge_index, edge_time, x, return_embeddings=False):
            
        src, dst = edge_index[0], edge_index[1]  # Extract sender & receiver nodes

        # Initialize x with real node features
        x = self.memory.memory.detach() + x  # Combine memory states and real metadata embeddings

        # Extract raw messages (Use stored memory as messages)
        raw_msg = x[src]

        self.memory.update_state(
            src=src, 
            dst=dst, 
            t=edge_time.squeeze(-1).long(),  # Convert (num_edges, 1) → (num_edges,)
            raw_msg=raw_msg
        )

        # Apply Transformer GCN
        x = self.conv(x, edge_index)

        # Extract user and movie embeddings
        user_embeddings = x[src]  # User embeddings
        movie_embeddings = x[dst]  # Movie embeddings

        # Concatenate user and movie embeddings per interaction
        interaction_embeddings = torch.cat([user_embeddings, movie_embeddings], dim=-1)

        if return_embeddings:
            projected_embeddings = self.projection(interaction_embeddings)  # Reduce to 32D
            return [{"userId": int(src[i].item()), "itemId": int(dst[i].item()), "embedding": projected_embeddings[i].tolist()} for i in range(len(src))]
        
        # Predict rating using MLP and scale to [1,5]
        rating = self.mlp(interaction_embeddings)
        rating = torch.sigmoid(rating) * 4 + 1  # Scale output to [1,5]
        
        return rating.squeeze(-1)  # Ensure correct shape

def extract_title_and_year(movie_title):
    match = re.search(r"^(.*)\s\((\d{4})\)$", movie_title)
    if match:
        title, year = match.group(1).strip(), int(match.group(2))
        return title, year
    print(f"❌ Error: Could not extract year from '{movie_title}'")
    return None, None

def clean_title(title):
    match = re.search(r"^(.*?)\s\((.*?)\)$", title)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return title, None

def fetch_movie_details(movie_title, release_year):
    global ia  # Use global instance
    search_titles = [movie_title]

    # Handle "The Movie_Title" variations
    if "," in movie_title and "The" in movie_title.split()[-1]:
        search_titles.append("The " + movie_title.rsplit(", ", 1)[0])

    # Remove alternative titles in parentheses unless necessary
    main_title, alt_title = clean_title(movie_title)
    if main_title not in search_titles:
        search_titles.append(main_title)
    if alt_title and alt_title not in search_titles:
        search_titles.append(alt_title)

    matched_movie = None
    for search_title in search_titles:
        try:
            search_results = ia.search_movie(search_title)
            if search_results:
                for movie in search_results:
                    if 'year' in movie.keys() and movie['year'] in [release_year, release_year - 1, release_year + 1]:
                        matched_movie = movie
                        break
            if matched_movie:
                break
        except Exception as e:
            print(f"❌ IMDb API Error for '{search_title}': {e}")

    if not matched_movie:
        print(f"❌ Error: No exact match for '{movie_title} ({release_year})'")
        return None

    try:
        ia.update(matched_movie)
        return {
            "title": matched_movie.get('title'),
            "year": matched_movie.get('year'),
            "directors": [person['name'] for person in matched_movie.get('directors', [])],
            "cast": [person['name'] for person in matched_movie.get('cast', [])[:10]],
            "plot": matched_movie.get('plot outline', None),
            "imdb_rating": matched_movie.get('rating', None),
            "number_of_votes": matched_movie.get('votes', None),
            "imdb_url": f"https://www.imdb.com/title/tt{matched_movie.movieID}/",
            "previous_records": [{
                "fetched_date": datetime.today().strftime("%d/%m/%Y"),
                "number_of_votes": matched_movie.get('votes', None),
                "imdb_rating": matched_movie.get('rating', None)
            }]
        }
    except Exception as e:
        print(f"❌ Failed to fetch data for '{movie_title} ({release_year})': {e}")
        return None

def process_movie_for_update(movie_id, movie_data, vote_change_threshold, rating_change_threshold):
    """Fetch IMDb details using the stored IMDb URL and update if conditions are met."""
    if "imdb_url" not in movie_data or not movie_data["imdb_url"]:
        print(f"❌ Skipping {movie_data['title']} (ID: {movie_id}) - No IMDb URL found.")
        return None  # Skip movies with missing IMDb URLs

    # Extract IMDb movie ID from the stored URL
    match = re.search(r'tt(\d+)', movie_data["imdb_url"])
    if not match:
        print(f"❌ Skipping {movie_data['title']} (ID: {movie_id}) - Invalid IMDb URL format.")
        return None

    imdb_movie_id = match.group(1)
    
    try:
        # Fetch movie data from IMDb using the extracted IMDb ID
        movie = ia.get_movie(imdb_movie_id)
        ia.update(movie, info=['vote details', 'main'])  # imdbpy has changed and doesn't show votes by default!
        
        if not movie:
            print(f"❌ IMDb API Error: Could not fetch movie data for {movie_data['title']} (ID: {movie_id})")
            return None

        updated_data = {
            "number_of_votes": movie.get("votes"),
            "imdb_rating": movie.get("rating")
        }

        if updated_data["number_of_votes"] is None or updated_data["imdb_rating"] is None:
            print(f"⚠️ Skipping update: {movie_data['title']} (ID: {movie_id}) - Missing IMDb data.")
            return None  # Skip if data is missing

        # Ensure previous_records exists
        if "previous_records" not in movie_data or not movie_data["previous_records"]:
            movie_data["previous_records"] = []

        last_record = movie_data["previous_records"][-1] if movie_data["previous_records"] else {}

        should_update = False

        # Check for significant change in votes or rating
        if "number_of_votes" in last_record and last_record["number_of_votes"] is not None:
            if updated_data["number_of_votes"] - last_record["number_of_votes"] >= vote_change_threshold:
                should_update = True

        if "imdb_rating" in last_record and last_record["imdb_rating"] is not None:
            if abs(updated_data["imdb_rating"] - last_record["imdb_rating"]) >= rating_change_threshold:
                should_update = True

        if should_update:
            # Append new record
            movie_data["previous_records"].append({
                "fetched_date": datetime.today().strftime("%d/%m/%Y"),
                "number_of_votes": updated_data["number_of_votes"],
                "imdb_rating": updated_data["imdb_rating"]
            })

            # Update latest values
            movie_data["number_of_votes"] = updated_data["number_of_votes"]
            movie_data["imdb_rating"] = updated_data["imdb_rating"]

            print(f"✅ Updated {movie_data['title']} (ID: {movie_id})")
            return movie_id, movie_data

        return None  # No significant changes

    except Exception as e:
        print(f"❌ Error updating {movie_data['title']} (ID: {movie_id}): {e}")
        return None

def process_movie_for_initial_fetch(movie_id, full_title):
    movie_title, release_year = extract_title_and_year(full_title)

    if movie_title and release_year:
        movie_data = fetch_movie_details(movie_title, release_year)
        if movie_data:
            return movie_id, movie_data
        else:
            #Log failed movies in JSON using `movie_id` as key
            failed_movies_path = "//FAILED_MOVIES.json"
            
            #Load existing failed movies to avoid overwriting
            if os.path.exists(failed_movies_path):
                with open(failed_movies_path, "r", encoding="utf-8") as f:
                    failed_movies = json.load(f)
            else:
                failed_movies = {}

            #Store failed movie under its `movie_id`
            failed_movies[movie_id] = {"title": movie_title, "year": release_year}

            #Save updated failed movies back to JSON
            with open(failed_movies_path, "w", encoding="utf-8") as f:
                json.dump(failed_movies, f, indent=4, ensure_ascii=False)

            print(f"🚨 Logged failed movie: {movie_title} ({release_year}) with ID {movie_id}")
    
    return None

def compute_popularity_momentum(movie_data):
    """Calculate the change in rating and votes over time to track momentum."""
    if len(movie_data.get("previous_records", [])) < 2:
        return 0.0  # Default value for insufficient data

    prev = movie_data["previous_records"][-2]
    latest = movie_data["previous_records"][-1]

    vote_change = (latest["number_of_votes"] or 1) - (prev["number_of_votes"] or 1)
    rating_change = (latest["imdb_rating"] or 5.0) - (prev["imdb_rating"] or 5.0)

    return (rating_change * 10) + (vote_change / 1000)  # Weighted combination

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val) if value is not None else 0.5  # Default 0.5

def create_graph(ratings_subset, user_features, movie_features):
    edge_index = torch.tensor(np.array([ratings_subset["user_id"].values, ratings_subset["movie_id"].values]), dtype=torch.long)
    edge_attr = torch.tensor(
        np.column_stack([ratings_subset["rating"].values, ratings_subset["popularity_momentum"].values]), 
        dtype=torch.float32
    )
    edge_time = torch.tensor(ratings_subset["timestamp"].values, dtype=torch.float32).view(-1, 1)
    
    node_features = torch.cat([user_features, movie_features], dim=0)  # Combine user/movie embeddings
    
    return Data(
        x=node_features,  # Node features (Users + Movies)
        edge_index=edge_index,  # User-Movie Interactions
        edge_attr=edge_attr,  # Rating values
        edge_time=edge_time  # Timestamps
    )

def evaluate_model(model, test_graph, K):
    """Evaluate the model on the test dataset with a dynamically set K (no logging)."""
    model.eval()
    
    with torch.no_grad():
        predicted_ratings = model(test_graph.edge_index, test_graph.edge_time, test_graph.x, return_embeddings=False)
        true_ratings = test_graph.edge_attr[:, 0]  # Actual ratings
    
    # ✅ Compute RMSE and MAE
    rmse = np.sqrt(mean_squared_error(true_ratings.cpu().numpy(), predicted_ratings.cpu().numpy()))
    mae = mean_absolute_error(true_ratings.cpu().numpy(), predicted_ratings.cpu().numpy())

    # ✅ Compute ranking-based metrics
    #precision, recall, ndcg, hit_ratio = compute_ranking_metrics(predicted_ratings, true_ratings, K)
    precision, recall, ndcg, hit_ratio = compute_ranking_metrics(predicted_ratings, true_ratings, K, test_graph.edge_index)

    # ✅ Compute coverage
    unique_recommended_items = torch.unique(test_graph.edge_index[1]).numel()
    total_items = test_graph.x.shape[0]
    coverage = unique_recommended_items / total_items

    return {
        "RMSE": rmse,
        "MAE": mae,
        "Precision@K": precision,
        "Recall@K": recall,
        "NDCG@K": ndcg,
        "Hit Ratio@K": hit_ratio,
        "Coverage": coverage
    }

def compute_ranking_metrics(pred_ratings, true_ratings, K, edge_index):
    """Compute Precision@K, Recall@K, NDCG@K, and Hit Ratio@K based on Leave-One-Out evaluation."""

    users = edge_index[0].unique()  # Unique users in test set
    hits = 0  # Number of users for which the true item is in Top-K
    total_users = len(users)

    precision_sum = 0.0
    recall_sum = 0.0
    ndcg_sum = 0.0

    for user in users:
        # Get indices of all items this user interacted with
        user_indices = (edge_index[0] == user).nonzero(as_tuple=True)[0]

        if len(user_indices) == 0:
            continue  # Skip users with no interactions (shouldn't happen in LOO)

        # Extract user's predicted ratings and true ratings
        user_pred_ratings = pred_ratings[user_indices]
        user_true_ratings = true_ratings[user_indices]

        # Get the index of the **ground-truth item** (LOO means there’s only one)
        ground_truth_index = torch.argmax(user_true_ratings).item()

        # Sort items by predicted score
        sorted_indices = torch.argsort(user_pred_ratings, descending=True)

        # Select Top-K items
        top_k_indices = sorted_indices[:K]

        # Compute HR@K (1 if ground truth is in top-K, else 0)
        hit = 1 if ground_truth_index in top_k_indices else 0
        hits += hit

        # Compute Precision@K
        relevant_items = (user_true_ratings >= 0.6).float()  # Items rated >3.5 are relevant
        retrieved_relevant = relevant_items[top_k_indices].sum().item()
        precision = retrieved_relevant / K if K > 0 else 0
        precision_sum += precision

        # Compute Recall@K (Same as HR@K in LOO)
        recall = hit
        recall_sum += recall

        # Compute NDCG@K
        gains = (2 ** user_true_ratings[top_k_indices] - 1)  # DCG formula
        discounts = torch.log2(torch.arange(1, K + 1, dtype=torch.float32) + 1)
        dcg = (gains / discounts).sum().item()

        ideal_gains = (2 ** torch.sort(user_true_ratings, descending=True).values[:K] - 1)
        ideal_dcg = (ideal_gains / discounts).sum().item()
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
        ndcg_sum += ndcg

    # Normalize by the number of users
    hr = hits / total_users if total_users > 0 else 0
    precision = precision_sum / total_users if total_users > 0 else 0
    recall = recall_sum / total_users if total_users > 0 else 0
    ndcg = ndcg_sum / total_users if total_users > 0 else 0

    return precision, recall, ndcg, hr

def main():
    UPDATE_FLAG = False
    VOTE_CHANGE_THRESHOLD = 50
    RATING_CHANGE_THRESHOLD = 0.1
    NUM_DIRECTORS = 1200
    NUM_ACTORS = 11000
    GENRE_DIM = 8
    DIRECTOR_DIM = 16
    ACTOR_DIM = 32
    PLOT_DIM = 64
    HIDDEN_DIM = 128
    OUT_DIM = 32  # This will be our final embedding size
    EPOCHS = 11
    LEARNING_RATE = 0.0005
    L2_LAMBDA = 0.0001
    METRIC_K = 10
    TOTAL_EMBEDDING_DIM = GENRE_DIM + DIRECTOR_DIM + ACTOR_DIM + PLOT_DIM + 8  # Final size

    # Define file paths
    U_DATA_PATH = "//dataset//u.data"
    U_ITEM_PATH = "//dataset//u.item"
    IMDB_METADATA_PATH = "//IMDB_METADATA.json"
    IMDB_METADATA_WITH_EMBEDDINGS_PATH = "//IMDB_METADATA_WITH_EMBEDDINGS.json"
        
    # Start time tracking
    start_time = time.time()
    start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Readable start time
    print(f"\n⏳ STARTED: {start_timestamp}\n")

    # Define genre columns in MovieLens dataset
    GENRE_COLUMNS = [
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
        "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    
    ratings_df = pd.read_csv(U_DATA_PATH, sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])

    # Normalize Ratings (1-5 → 0-1)
    ratings_df["rating"] = ratings_df["rating"] / 5.0
    
    # Convert User & Movie IDs to Zero-Based Index
    ratings_df["user_id"] -= 1
    ratings_df["movie_id"] -= 1

    # Check if the JSON file exists
    if not os.path.exists(IMDB_METADATA_PATH):
        print("🔍 IMDB_METADATA JSON file not found. Creating a new file...")

        # Read the MovieLens u.item dataset
        movies_df = pd.read_csv(U_ITEM_PATH, sep="|", encoding="latin-1", header=None)

        # Define correct column names
        column_names = ["movie_id", "title", "release_date", "video_release_date", "IMDb_URL"] + GENRE_COLUMNS

        # Assign column names to DataFrame
        movies_df.columns = column_names

        # Convert movie_id to string
        movies_df["movie_id"] = movies_df["movie_id"].astype(str)

        genre_mapping = movies_df.set_index("movie_id")[GENRE_COLUMNS].to_dict(orient="index")

        movie_details_list = {}

        # Parallel processing for initial movie fetch
        print("🔍 Fetching IMDb data in parallel...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_movie_for_initial_fetch, row["movie_id"], row["title"].strip()): row["movie_id"] for _, row in movies_df.iterrows()}

            for future in futures:
                result = future.result()
                if result:
                    movie_id, movie_data = result
                    movie_data["genres"] = list(genre_mapping[movie_id].values())
                    movie_details_list[movie_id] = movie_data

        with open(IMDB_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(movie_details_list, f, indent=4, ensure_ascii=False)
        print(f"\n✅ New JSON file created at '{IMDB_METADATA_PATH}'.")

    else:
        print("🔍 IMDB_METADATA JSON file found. Checking if updates are needed...")

        with open(IMDB_METADATA_PATH, "r", encoding="utf-8") as f:
            movie_details_list = json.load(f)

        today = datetime.today()
        
        # Step 1: Find the latest fetched_date in all objects
        all_dates = []
        for movie_data in movie_details_list.values():
            if "previous_records" in movie_data and movie_data["previous_records"]:
                for record in movie_data["previous_records"]:
                    try:
                        date_obj = datetime.strptime(record["fetched_date"], "%d/%m/%Y")
                        all_dates.append(date_obj)
                    except ValueError:
                        print(f"⚠️ Invalid date format: {record['fetched_date']} in {movie_data['title']}")

        # Step 2: Get the maximum date (latest update across all movies)
        latest_fetched_date = max(all_dates) if all_dates else None

        # Step 3: Compare with today's date
        if latest_fetched_date and (today - latest_fetched_date).days >= 7:
            print(f"🔍 More than a week has passed since the last update ({latest_fetched_date.strftime('%d/%m/%Y')}). Checking for updates...")
            UPDATE_FLAG = True
            # Parallel processing for updates
            print("🔍 Starting parallel IMDb updates...")
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(process_movie_for_update, movie_id, movie_data, VOTE_CHANGE_THRESHOLD, RATING_CHANGE_THRESHOLD): movie_id for movie_id, movie_data in movie_details_list.items()}

                for future in futures:
                    result = future.result()
                    if result:
                        movie_id, updated_data = result
                        movie_details_list[movie_id] = updated_data

            with open(IMDB_METADATA_PATH, "w", encoding="utf-8") as f:
                json.dump(movie_details_list, f, indent=4, ensure_ascii=False)
            print(f"\n✅ JSON file updated at '{IMDB_METADATA_PATH}'.")
        else:
            print("Less than a week since last update. No updates needed.")

    if not os.path.exists(IMDB_METADATA_WITH_EMBEDDINGS_PATH) or UPDATE_FLAG == True:
        print("🔍 IMDB_METADATA_WITH_EMBEDDINGS JSON file not found or needs updating. Creating a new file...")

        # Define Embedding Layers
        genre_fc = nn.Linear(19, GENRE_DIM)
        director_embedding = nn.Embedding(NUM_DIRECTORS, DIRECTOR_DIM)
        actor_embedding = nn.Embedding(NUM_ACTORS, ACTOR_DIM)

        # Load Pretrained BERT Model
        bert_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Convert Movies into Embeddings
        for movie_id, movie in movie_details_list.items():
            # Handle Missing Values
            genres = movie.get("genres", [0] * 19)  # Default to empty one-hot vector
            directors = movie.get("directors", [])
            cast = movie.get("cast", [])
            plot_text = movie.get("plot", "")  # Use empty string if no plot
            imdb_rating = movie.get("imdb_rating", 5.0)  # Default rating
            num_votes = movie.get("number_of_votes", 1)  # Default votes
            release_year = movie.get("year", 1995)  # Default mid-range year

            # 1️⃣ Convert Genre One-Hot to Dense Embedding
            genre_vector = torch.tensor(genres, dtype=torch.float32)
            genre_embedding_out = genre_fc(genre_vector)

            # 2️⃣ Convert Director ID to Embedding
            director_idx = hash(directors[0]) % NUM_DIRECTORS if directors else 0
            director_embedding_out = director_embedding(torch.tensor(director_idx))

            # 3️⃣ Convert Multiple Actors to a Single Embedding
            actor_ids = [hash(actor) % NUM_ACTORS for actor in cast[:5]]  # Consider up to 5 actors
            if actor_ids:
                actor_embeddings = actor_embedding(torch.tensor(actor_ids))  # Get embeddings for all 5 actors
                actor_embedding_out = actor_embeddings.mean(dim=0)  # Average actor embeddings
            else:
                actor_embedding_out = torch.zeros(ACTOR_DIM)  # Default zero vector if no actors

            # 4️⃣ Convert Plot to NLP-Based Embedding (Real BERT Model)
            plot_embedding_out = torch.tensor(bert_model.encode(plot_text)) if plot_text else torch.zeros(PLOT_DIM)
            plot_embedding_out = plot_embedding_out[:PLOT_DIM]  # Reduce to 64D

            # 5️⃣ Normalize Numerical Values
            norm_rating = normalize(imdb_rating, 1, 10)
            norm_votes = normalize(num_votes, 1, 4_000_000)
            norm_year = normalize(release_year, 1922, 1999)

            # 6️⃣ Compute Popularity Momentum
            popularity_momentum = compute_popularity_momentum(movie)

            # 7️⃣ Concatenate All Features into One Embedding
            movie_embedding = torch.cat([
                genre_embedding_out,  # (8D)
                director_embedding_out,  # (16D)
                actor_embedding_out,  # (32D)
                plot_embedding_out,  # (64D)
                torch.tensor([norm_rating, norm_votes, norm_year, popularity_momentum]),  # (4D)
                torch.zeros(4)  # 🔹 Add 4 zero-padded dimensions
            ], dim=-1)

            # Convert to List and Store in JSON
            movie["embedding"] = movie_embedding.tolist()

        # Save Updated JSON
        with open(IMDB_METADATA_WITH_EMBEDDINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(movie_details_list, f, indent=4, ensure_ascii=False)

        print(f"✅ Movie embeddings added and saved to {IMDB_METADATA_WITH_EMBEDDINGS_PATH}")
    
    else:
        print("🔍 IMDB_METADATA_WITH_EMBEDDINGS JSON file found and no updates were needed.")

        with open(IMDB_METADATA_WITH_EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
            movie_details_list = json.load(f)

    # Compute normalized popularity momentum for each movie
    popularity_momentum_dict = {movie_id: compute_popularity_momentum(data) for movie_id, data in movie_details_list.items()}
    ratings_df["popularity_momentum"] = ratings_df["movie_id"].map(popularity_momentum_dict)
    
    # Extract Unique Users
    unique_users = ratings_df["user_id"].unique()

    # Split Users into Train/Test (80/20)
    train_users, test_users = train_test_split(unique_users, test_size=0.2, random_state=42)

    # Filter Ratings Data for Train & Test
    train_ratings = ratings_df[ratings_df["user_id"].isin(train_users)]
    test_ratings = ratings_df[ratings_df["user_id"].isin(test_users)]

    # Convert JSON to Tensor (Same for Both)
    num_movies = len(movie_details_list)
    embedding_dim = len(movie_details_list["1"]["embedding"])

    movie_features = torch.zeros((num_movies, embedding_dim))
    for movie_id, data in movie_details_list.items():
        idx = int(movie_id) - 1  # Convert movie_id to zero-based index
        movie_features[idx] = torch.tensor(data["embedding"], dtype=torch.float32)

    num_users = ratings_df["user_id"].max() + 1  # Total number of users
    torch.manual_seed(42)  # Ensures reproducibility
    user_features = torch.rand((num_users, embedding_dim))
    #user_features = torch.zeros((num_users, embedding_dim))  # Initialize with zeros
    
    train_graph = create_graph(train_ratings, user_features, movie_features)
    test_graph = create_graph(test_ratings, user_features, movie_features)

    # Save Train & Test Graphs
    torch.save(train_graph, "train_graph.pt")
    torch.save(test_graph, "test_graph.pt")
    print("✅ Train & Test Graphs Created!")
    
    # Get total number of nodes (users + movies)
    num_nodes = train_graph.x.shape[0]

    model = TGNRecommender(
        in_channels=TOTAL_EMBEDDING_DIM,
        hidden_channels=HIDDEN_DIM,
        out_channels=OUT_DIM,
        num_nodes=num_nodes
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_LAMBDA)
    loss_fn = torch.nn.MSELoss()
    
    print("\n🚀 Starting Training...\n")
    for epoch in range(EPOCHS):
        model.train()
        torch.autograd.set_detect_anomaly(True)

        optimizer.zero_grad()

        # Forward pass on train data
        predicted_ratings = model(train_graph.edge_index, train_graph.edge_time, train_graph.x, return_embeddings=False)


        # Compute loss (MSE Loss for predicted ratings)
        loss = loss_fn(predicted_ratings, train_graph.edge_attr[:, 0])  # Compare with actual ratings

        # Backpropagation
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item():.4f}")

    print("\n✅ Training Completed!\n")
    
    train_embeddings = model(train_graph.edge_index, train_graph.edge_time, train_graph.x, return_embeddings=True)
    
    # Convert IDs back to original MovieLens values (1-based indexing)
    for entry in train_embeddings:
        entry["userId"] += 1
        entry["itemId"] += 1

    # Save JSON
    with open("//train_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(train_embeddings, f, indent=4, ensure_ascii=False)

    print("✅ Train embeddings saved (last epoch)!")
    
    print("\n🚀 Starting Testing...\n")
    
    results = evaluate_model(model, test_graph, METRIC_K)

    # Print results
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    print("\n✅ Testing Completed!\n")
    
    test_embeddings = model(test_graph.edge_index, test_graph.edge_time, test_graph.x, return_embeddings=True)

    # Convert IDs back to original MovieLens values (1-based indexing)
    for entry in test_embeddings:
        entry["userId"] += 1
        entry["itemId"] += 1

    # Save JSON
    with open("//test_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(test_embeddings, f, indent=4, ensure_ascii=False)

    print("✅ Test embeddings saved!")

    # End time tracking
    end_time = time.time()
    end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Readable end time
    execution_time = end_time - start_time
    hours, rem = divmod(execution_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Print completion details
    print(f"\n✅ COMPLETED: {end_timestamp}")
    print(f"⏳ Execution Time: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds.\n")
    
if __name__ == '__main__':
    main()
