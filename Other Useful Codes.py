########################################################## Getting Info About Dataset ##########################################################
import json

# ✅ Load the JSON file containing movie metadata
json_file_path = "//movielens_imdb_data.json"

with open(json_file_path, "r", encoding="utf-8") as f:
    movie_data = json.load(f)

# ✅ Extract values
all_ratings = []
all_votes = []
all_years = []
all_directors = set()
all_actors = set()

for movie in movie_data.values():
    if movie["imdb_rating"] is not None:
        all_ratings.append(movie["imdb_rating"])
    if movie["number_of_votes"] is not None:
        all_votes.append(movie["number_of_votes"])
    if movie["year"] is not None:
        all_years.append(movie["year"])
    if movie["directors"]:
        all_directors.update(movie["directors"])
    if movie["cast"]:
        all_actors.update(movie["cast"])

# ✅ Calculate min & max values
min_rating, max_rating = min(all_ratings), max(all_ratings)
min_votes, max_votes = min(all_votes), max(all_votes)
min_year, max_year = min(all_years), max(all_years)

# ✅ Count unique directors and actors
num_unique_directors = len(all_directors)
num_unique_actors = len(all_actors)

# ✅ Print results
print(f"📌 IMDb Rating: Min = {min_rating}, Max = {max_rating}")
print(f"📌 Number of Votes: Min = {min_votes}, Max = {max_votes}")
print(f"📌 Year: Min = {min_year}, Max = {max_year}")
print(f"📌 Unique Directors: {num_unique_directors}")
print(f"📌 Unique Actors: {num_unique_actors}")
