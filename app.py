# -------------------------
# IMPORTS
# -------------------------
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tmdbv3api import TMDb, Movie
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------
# TMDB SETUP
# -------------------------
tmdb = TMDb()
tmdb.api_key = "98a3c31c5eb0f7b32969a83641aa798e"  # Replace with your TMDB API key
tmdb.language = 'en'
tmdb_movie = Movie()
poster_cache = {}

def placeholder_poster(width=150, height=225, color=(200, 200, 200)):
    img = np.full((height, width, 3), color, dtype=np.uint8)
    return Image.fromarray(img)

def clean_title(title):
    return re.sub(r'\(\d{4}\)', '', title).strip()

def fetch_poster_tmdb(title):
    cleaned_title = clean_title(title)
    if cleaned_title in poster_cache:
        return poster_cache[cleaned_title]
    try:
        search = tmdb_movie.search(cleaned_title)
        if search and len(search) > 0 and search[0].poster_path:
            url = f"https://image.tmdb.org/t/p/w500{search[0].poster_path}"
            poster_cache[cleaned_title] = url
            return url
    except Exception as e:
        print(f"TMDB fetch error for {title}: {e}")
    poster_cache[cleaned_title] = placeholder_poster()
    return poster_cache[cleaned_title]

def fetch_posters_parallel(titles):
    results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_title = {executor.submit(fetch_poster_tmdb, t): t for t in titles}
        for future in as_completed(future_to_title):
            title = future_to_title[future]
            try:
                results[title] = future.result()
            except Exception:
                results[title] = placeholder_poster()
    return results

# -------------------------
# LOAD DATA & MODELS
# -------------------------
movies = pd.read_csv("movies.csv")
cosine_sim = np.load("cosine_sim.pkl", allow_pickle=True)

with open("user_enc.pkl", "rb") as f:
    user_enc = pickle.load(f)
with open("movie_enc.pkl", "rb") as f:
    movie_enc = pickle.load(f)

final_model = load_model("final_model.h5")

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def match_title(title):
    title = re.escape(title)
    matches = movies[movies['title'].str.contains(title, case=False, na=False)]
    return matches.iloc[0]['title'] if len(matches) > 0 else None

@st.cache_data
def get_content_scores(title):
    matched_title = match_title(title)
    if matched_title is None:
        return None
    idx = movies[movies['title'] == matched_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    results = [(movies.iloc[i]['movieId'], score) for i, score in sim_scores]
    return results

def content_based_top_k(title, top_k=5):
    scores = get_content_scores(title)
    if scores is None:
        return None
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[1] < 1.0][:top_k]
    results = []
    for mid, score in scores:
        movie_title = movies[movies['movieId'] == mid].iloc[0]['title']
        results.append((movie_title, float(score)))
    return results

def predict_user_movie_rating(user_id, movie_id):
    u = user_enc.transform([user_id])[0]
    m = movie_enc.transform([movie_id])[0]
    u = u.reshape(1, 1)
    m = m.reshape(1, 1)
    pred = final_model.predict([u, m], verbose=0)[0][0]
    return float(pred)

def hybrid_recommend_fast(user_id, movie_title, top_k=10, alpha=0.7, max_candidates=30, exclude_ids=[]):
    content_scores = get_content_scores(movie_title)
    if content_scores is None:
        return []
    
    content_scores = sorted(content_scores, key=lambda x: x[1], reverse=True)
    content_scores = [s for s in content_scores if s[1] < 1.0][:max_candidates]

    movie_ids, content_sims = zip(*content_scores)
    movie_ids = np.array(movie_ids)
    content_sims = np.array(content_sims)

    mask = np.isin(movie_ids, exclude_ids, invert=True)
    movie_ids = movie_ids[mask]
    content_sims = content_sims[mask]

    if len(movie_ids) == 0:
        return []

    u = user_enc.transform([user_id])[0]
    users = np.full(len(movie_ids), u)
    m = movie_enc.transform(movie_ids)
    users = users.reshape(-1, 1)
    m = m.reshape(-1, 1)

    cf_scores = final_model.predict([users, m], verbose=0).flatten()
    hybrid_scores = alpha * cf_scores + (1 - alpha) * content_sims

    results = []
    for mid, score in zip(movie_ids, hybrid_scores):
        movie_title_candidate = movies[movies['movieId'] == mid].iloc[0]['title']
        results.append((mid, movie_title_candidate, float(score)))

    results = sorted(results, key=lambda x: x[2], reverse=True)
    return results[:top_k]

# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")
st.title("🎬 Hybrid Movie Recommendation System")

# -------------------------
# USER INPUTS
# -------------------------
col1, col2 = st.columns([1, 4])

with col1:
    user_id = st.number_input(
        "User ID",
        min_value=1,
        max_value=1000,
        value=1,
        step=1,
        format="%d",
        help="Enter your user ID"
    )

with col2:
    movie_titles = movies['title'].tolist()
    selected_movie = st.selectbox(
        "Search for a movie",
        options=[""] + movie_titles,
        index=0,
        format_func=lambda x: "Type here to search..." if x == "" else x,
        help="Start typing to see suggestions..."
    )

recommend_button = st.button("🔍 Recommend")

# -------------------------
# ON BUTTON CLICK
# -------------------------
if recommend_button and selected_movie != "":
    matched_title = match_title(selected_movie)
    if matched_title is None:
        st.warning("No movie found with that name.")
    else:
        st.subheader(f"🎞️ Search result: **{matched_title}**")

        # -------------------------
        # Content-based
        # -------------------------
        similar_movies = content_based_top_k(matched_title)
        similar_ids = [movies[movies['title']==title].iloc[0]['movieId'] for title, _ in similar_movies] if similar_movies else []

        # -------------------------
        # Hybrid
        # -------------------------
        hybrid_movies = hybrid_recommend_fast(user_id, matched_title, top_k=10, exclude_ids=similar_ids)

        # Fetch posters in parallel
        all_titles = [title for title, _ in similar_movies] + [title for _, title, _ in hybrid_movies]
        poster_dict = fetch_posters_parallel(all_titles)

        # -------------------------
        # Display Content-based (5 movies, single row)
        # -------------------------
        st.markdown("## 🔍 Similar Movies (Content-based)")
        if similar_movies:
            cols = st.columns(5)
            for j, (title, score) in enumerate(similar_movies):
                with cols[j]:
                    poster = poster_dict.get(title, placeholder_poster())
                    st.image(poster, width=120)
                    st.write(f"**{title}**")
                    st.caption(f"Similarity: {score:.3f}")

        # -------------------------
        # Display Hybrid (10 movies, 5 per row)
        # -------------------------
        st.markdown("---")
        st.markdown("## ⭐ You May Also Like (Hybrid)")
        if hybrid_movies:
            for i in range(0, 10, 5):
                cols2 = st.columns(5)
                for j, (_, title, score) in enumerate(hybrid_movies[i:i+5]):
                    with cols2[j]:
                        poster = poster_dict.get(title, placeholder_poster())
                        st.image(poster, width=120)
                        st.write(f"**{title}**")
                        st.caption(f"Score: {score:.3f}")
