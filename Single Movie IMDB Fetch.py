import re
import os
import json
from imdb import IMDb
from datetime import datetime

ia = IMDb()

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

def process_movie_for_initial_fetch(movie_id, full_title):
    movie_title, release_year = extract_title_and_year(full_title)
    
    if movie_title and release_year:
        movie_data = fetch_movie_details(movie_title, release_year)
        if movie_data:
            return movie_id, movie_data
    return None


movie_id = 374
title = "French Twist (Gazon maudit) (1995)"

result = process_movie_for_initial_fetch(movie_id, title)

if result:
    movie_id, movie_data = result
    print(movie_data)