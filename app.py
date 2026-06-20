import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie & TV Recommender", page_icon="🎬", layout="wide")

API_KEY  = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI4YzI3OGY5MDRjMTVkMzY2M2RmYjIzNDc4ODhiZmUzYiIsIm5iZiI6MTc4MDU2MzQ0Ni4zNDYsInN1YiI6IjZhMjEzZGY2Mjc5NzUzZDNmYzI2ZDU1NyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.tJebKB3hiovMT7aOOSyNa40WIZEWCugkSc8ga0LdAdo"
HEADERS  = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}
BASE_URL = "https://api.themoviedb.org/3"
IMG_URL  = "https://image.tmdb.org/t/p/w300"


@st.cache_data(show_spinner="Fetching data from TMDB...")
def fetch_items(mode="movie", pages=50):
    items = []
    genre_map = {}

    r = requests.get(f"{BASE_URL}/genre/{mode}/list", headers=HEADERS)
    for g in r.json().get("genres", []):
        genre_map[g["id"]] = g["name"]

    title_key = "title" if mode == "movie" else "name"
    date_key  = "release_date" if mode == "movie" else "first_air_date"

    for endpoint in ["popular", "top_rated"]:
        for page in range(1, pages + 1):
            r = requests.get(
                f"{BASE_URL}/{mode}/{endpoint}",
                headers=HEADERS,
                params={"language": "en-US", "page": page}
            )
            if r.status_code != 200:
                break
            for m in r.json().get("results", []):
                title = m.get(title_key, "")
                if not title:
                    continue
                genres = " ".join([genre_map.get(gid, "") for gid in m.get("genre_ids", [])])
                items.append({
                    "title":    title,
                    "overview": m.get("overview", ""),
                    "genres":   genres,
                    "rating":   m.get("vote_average", 0),
                    "votes":    m.get("vote_count", 0),
                    "poster":   IMG_URL + m["poster_path"] if m.get("poster_path") else None,
                    "year":     m.get(date_key, "")[:4],
                    "combined": genres + " " + m.get("overview", ""),
                })

    df = pd.DataFrame(items).drop_duplicates(subset="title").reset_index(drop=True)
    return df


@st.cache_data(show_spinner="Building recommendation engine...")
def build_sim(df):
    tfidf  = TfidfVectorizer(stop_words="english", max_features=10000)
    matrix = tfidf.fit_transform(df["combined"].fillna(""))
    return cosine_similarity(matrix, matrix)


def recommend(query, df, sim, n=10):
    query_lower = query.lower().strip()
    matches = df[df["title"].str.lower().str.contains(query_lower, na=False)]
    if matches.empty:
        return None, None
    idx    = matches.index[0]
    scores = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)[1:n+1]
    return df.iloc[[i for i, _ in scores]], df.iloc[idx]


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Movie & TV Show Recommender")
st.markdown("Search any movie or TV show and get instant recommendations.")
st.markdown("---")

mode_label = st.radio("What are you looking for?", ["Movies", "TV Shows"], horizontal=True)
mode = "movie" if mode_label == "Movies" else "tv"

df  = fetch_items(mode=mode, pages=50)
sim = build_sim(df)

st.caption(f"Database loaded: {len(df):,} {mode_label.lower()}")
st.markdown("---")

search = st.text_input(
    f"Search a {mode_label[:-1].lower()} you like",
    placeholder="e.g. Inception, Squid Game, Breaking Bad, Avatar"
)

if search:
    recs, source = recommend(search, df, sim)
    if recs is None:
        st.error(f"Could not find '{search}'. Try another title.")

        # Show closest matches
        query_lower = search.lower()
        close = df[df["title"].str.lower().str.contains(query_lower[:4], na=False)]["title"].head(5).tolist()
        if close:
            st.markdown("**Did you mean:**")
            for t in close:
                st.markdown(f"- {t}")
    else:
        st.success(f"Because you liked **{source['title']} ({source['year']})**:")
        st.markdown("---")
        cols = st.columns(5)
        for i, (_, row) in enumerate(recs.head(10).iterrows()):
            with cols[i % 5]:
                if row["poster"]:
                    st.image(row["poster"], use_container_width=True)
                else:
                    st.markdown("_(No poster)_")
                st.markdown(f"**{row['title']}** ({row['year']})")
                st.markdown(f"⭐ {row['rating']:.1f}/10")
                st.caption(row["genres"])

st.markdown("---")

# Trending section
st.subheader(f"Top Rated {mode_label}")
top = df[df["votes"] > 500].sort_values("rating", ascending=False).head(10)
cols2 = st.columns(5)
for i, (_, row) in enumerate(top.iterrows()):
    with cols2[i % 5]:
        if row["poster"]:
            st.image(row["poster"], use_container_width=True)
        st.markdown(f"**{row['title']}** ({row['year']})")
        st.markdown(f"⭐ {row['rating']:.1f}/10")

st.markdown("---")
st.caption("Powered by TMDB API | TF-IDF + Cosine Similarity")