import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="movie_camera", layout="wide")

API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI4YzI3OGY5MDRjMTVkMzY2M2RmYjIzNDc4ODhiZmUzYiIsIm5iZiI6MTc4MDU2MzQ0Ni4zNDYsInN1YiI6IjZhMjEzZGY2Mjc5NzUzZDNmYzI2ZDU1NyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.tJebKB3hiovMT7aOOSyNa40WIZEWCugkSc8ga0LdAdo"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}
BASE_URL = "https://api.themoviedb.org/3"
IMG_URL  = "https://image.tmdb.org/t/p/w300"

@st.cache_data(show_spinner="Fetching movies from TMDB...")
def fetch_movies(pages=20):
    movies = []
    genre_map = {}

    # Get genre list
    r = requests.get(f"{BASE_URL}/genre/movie/list", headers=HEADERS)
    for g in r.json().get("genres", []):
        genre_map[g["id"]] = g["name"]

    for page in range(1, pages + 1):
        r = requests.get(f"{BASE_URL}/movie/popular", headers=HEADERS,
                         params={"language": "en-US", "page": page})
        for m in r.json().get("results", []):
            genres = " ".join([genre_map.get(gid, "") for gid in m.get("genre_ids", [])])
            movies.append({
                "id":       m["id"],
                "title":    m.get("title", ""),
                "overview": m.get("overview", ""),
                "genres":   genres,
                "rating":   m.get("vote_average", 0),
                "votes":    m.get("vote_count", 0),
                "poster":   IMG_URL + m["poster_path"] if m.get("poster_path") else None,
                "year":     m.get("release_date", "")[:4],
                "combined": genres + " " + m.get("overview", ""),
            })
    return pd.DataFrame(movies)


@st.cache_data(show_spinner="Building recommendation engine...")
def build_engine(df):
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = tfidf.fit_transform(df["combined"].fillna(""))
    sim = cosine_similarity(matrix, matrix)
    return sim


def recommend(title, df, sim, n=10):
    title_lower = title.lower().strip()
    matches = df[df["title"].str.lower().str.contains(title_lower)]
    if matches.empty:
        return None, None
    idx = matches.index[0]
    scores = list(enumerate(sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    rec_idx = [i[0] for i in scores]
    return df.iloc[rec_idx], df.iloc[idx]


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Movie Recommendation System")
st.markdown("Type a movie you like and get instant recommendations powered by TMDB.")

df  = fetch_movies(pages=20)
sim = build_engine(df)

st.markdown("---")

search = st.text_input("Search a movie you like", placeholder="e.g. Inception, Avatar, The Dark Knight")

if search:
    recs, source = recommend(search, df, sim)

    if recs is None:
        st.error(f"Could not find '{search}'. Try another title.")
    else:
        st.success(f"Because you liked **{source['title']} ({source['year']})**:")
        st.markdown("---")

        cols = st.columns(5)
        for i, (_, row) in enumerate(recs.head(10).iterrows()):
            with cols[i % 5]:
                if row["poster"]:
                    st.image(row["poster"], use_container_width=True)
                else:
                    st.markdown("No poster")
                st.markdown(f"**{row['title']}** ({row['year']})")
                st.markdown(f"Rating: {row['rating']:.1f}/10")
                st.markdown(f"*{row['genres']}*")

st.markdown("---")

# Trending section
st.subheader("Trending Movies")
top = df[df["votes"] > 1000].sort_values("rating", ascending=False).head(10)
cols2 = st.columns(5)
for i, (_, row) in enumerate(top.iterrows()):
    with cols2[i % 5]:
        if row["poster"]:
            st.image(row["poster"], use_container_width=True)
        st.markdown(f"**{row['title']}** ({row['year']})")
        st.markdown(f"Rating: {row['rating']:.1f}/10")

st.markdown("---")
st.caption("Powered by TMDB API | Content-based filtering using TF-IDF + Cosine Similarity")
