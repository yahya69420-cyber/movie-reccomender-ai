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


@st.cache_data(show_spinner="Fetching movies and TV shows from TMDB...")
def fetch_all(pages=30):
    items = []
    genre_map = {}

    for mode in ["movie", "tv"]:
        r = requests.get(f"{BASE_URL}/genre/{mode}/list", headers=HEADERS)
        for g in r.json().get("genres", []):
            genre_map[g["id"]] = g["name"]

    for mode in ["movie", "tv"]:
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
                        "id":       m["id"],
                        "title":    title,
                        "overview": m.get("overview", ""),
                        "genres":   genres,
                        "rating":   m.get("vote_average", 0),
                        "votes":    m.get("vote_count", 0),
                        "poster":   IMG_URL + m["poster_path"] if m.get("poster_path") else None,
                        "year":     m.get(date_key, "")[:4],
                        "type":     "Movie" if mode == "movie" else "TV Show",
                        "combined": genres + " " + m.get("overview", ""),
                    })

    df = pd.DataFrame(items).drop_duplicates(subset=["title","type"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner="Building recommendation engine...")
def build_sim(df):
    tfidf  = TfidfVectorizer(stop_words="english", max_features=10000)
    matrix = tfidf.fit_transform(df["combined"].fillna(""))
    return cosine_similarity(matrix, matrix)


def search_tmdb(query):
    r = requests.get(
        f"{BASE_URL}/search/multi",
        headers=HEADERS,
        params={"query": query, "language": "en-US", "page": 1}
    )
    return r.json().get("results", [])


def get_watch_providers(item_id, media_type):
    r = requests.get(
        f"{BASE_URL}/{media_type}/{item_id}/watch/providers",
        headers=HEADERS
    )
    data = r.json().get("results", {})
    us = data.get("US", data.get("GB", data.get("AE", {})))
    return us


def get_trailer(item_id, media_type):
    r = requests.get(
        f"{BASE_URL}/{media_type}/{item_id}/videos",
        headers=HEADERS,
        params={"language": "en-US"}
    )
    for v in r.json().get("results", []):
        if v.get("site") == "YouTube" and v.get("type") in ["Trailer", "Teaser"]:
            return f"https://www.youtube.com/watch?v={v['key']}"
    return None


def recommend(query, df, sim, n=10):
    query_lower = query.lower().strip()
    matches = df[df["title"].str.lower().str.contains(query_lower, na=False)]
    if matches.empty:
        return None, None
    idx    = matches.index[0]
    scores = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)[1:n+1]
    return df.iloc[[i for i, _ in scores]], df.iloc[idx]


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🎬 Movie & TV Show Recommender")
st.markdown("Search any movie or TV show — get recommendations and watch options instantly.")
st.markdown("---")

df  = fetch_all(pages=30)
sim = build_sim(df)

st.caption(f"Database: {len(df):,} titles (movies + TV shows combined)")

search = st.text_input(
    "Search a movie or TV show",
    placeholder="e.g. Squid Game, Inception, Breaking Bad, Avatar..."
)

if search:
    # First: live TMDB search for exact match
    results = search_tmdb(search)
    exact_match = None
    for r in results:
        if r.get("media_type") in ["movie", "tv"]:
            exact_match = r
            break

    if exact_match:
        media_type  = exact_match["media_type"]
        title       = exact_match.get("title") or exact_match.get("name", "")
        year        = (exact_match.get("release_date") or exact_match.get("first_air_date", ""))[:4]
        overview    = exact_match.get("overview", "No description available.")
        rating      = exact_match.get("vote_average", 0)
        poster_path = exact_match.get("poster_path")
        item_id     = exact_match["id"]
        label       = "Movie" if media_type == "movie" else "TV Show"

        st.markdown("---")
        col1, col2 = st.columns([1, 3])

        with col1:
            if poster_path:
                st.image(IMG_URL + poster_path, width=200)

        with col2:
            st.markdown(f"## {title} ({year})")
            st.markdown(f"**{label}** | ⭐ {rating:.1f}/10")
            st.markdown(overview)

            # Trailer
            trailer_url = get_trailer(item_id, media_type)
            if trailer_url:
                st.markdown(f"[▶ Watch Trailer on YouTube]({trailer_url})")

            # Watch providers
            providers = get_watch_providers(item_id, media_type)
            stream    = providers.get("flatrate", [])
            rent      = providers.get("rent", [])
            buy       = providers.get("buy", [])
            tmdb_link = providers.get("link", f"https://www.themoviedb.org/{media_type}/{item_id}")

            if stream:
                names = ", ".join([p["provider_name"] for p in stream[:5]])
                st.success(f"**Stream on:** {names}")
            elif rent:
                names = ", ".join([p["provider_name"] for p in rent[:5]])
                st.info(f"**Available to rent on:** {names}")
            elif buy:
                names = ", ".join([p["provider_name"] for p in buy[:3]])
                st.info(f"**Available to buy on:** {names}")
            else:
                st.warning("No streaming info available for your region.")

            st.markdown(f"[🔗 View on TMDB]({tmdb_link})")

        # Recommendations
        st.markdown("---")
        st.subheader(f"Because you searched **{title}** — you might also like:")

        recs, _ = recommend(title, df, sim)
        if recs is not None:
            cols = st.columns(5)
            for i, (_, row) in enumerate(recs.head(10).iterrows()):
                with cols[i % 5]:
                    if row["poster"]:
                        st.image(row["poster"], width=130)
                    rec_results = search_tmdb(row["title"])
                    rec_match = next((r for r in rec_results if r.get("media_type") in ["movie","tv"]), None)
                    if rec_match:
                        rec_id = rec_match["id"]
                        rec_type = rec_match["media_type"]
                        rec_link = f"https://www.themoviedb.org/{rec_type}/{rec_id}"
                        rec_trailer = get_trailer(rec_id, rec_type)
                        rec_providers = get_watch_providers(rec_id, rec_type)
                        stream = rec_providers.get("flatrate", [])
                        rec_stream = ", ".join([p["provider_name"] for p in stream[:3]]) if stream else None
                        st.markdown(f"**[{row['title']}]({rec_link})** ({row['year']})")
                        st.caption(f"{row['type']} | ⭐ {row['rating']:.1f}")
                        st.caption(row["genres"])
                        if rec_trailer:
                            st.markdown(f"[▶ Trailer]({rec_trailer})")
                        if rec_stream:
                            st.caption(f"Stream: {rec_stream}")
                    else:
                        st.markdown(f"**{row['title']}** ({row['year']})")
                        st.caption(f"{row['type']} | ⭐ {row['rating']:.1f}")
                        st.caption(row["genres"])
        else:
            # fallback to genre-based from TMDB search
            st.info("Showing similar titles from TMDB search.")
            for r2 in results[1:6]:
                t = r2.get("title") or r2.get("name", "")
                st.markdown(f"- {t}")

    else:
        st.error(f"Could not find '{search}'. Try a different spelling.")
        close = df[df["title"].str.lower().str.contains(search.lower()[:4], na=False)]["title"].head(5).tolist()
        if close:
            st.markdown("**Did you mean:**")
            for t in close:
                st.markdown(f"- {t}")

st.markdown("---")

# Trending
st.subheader("Top Rated Right Now")
top = df[df["votes"] > 1000].sort_values("rating", ascending=False).head(10)
cols2 = st.columns(5)
for i, (_, row) in enumerate(top.iterrows()):
    with cols2[i % 5]:
        if row["poster"]:
            st.image(row["poster"], width=130)
        rec_results2 = search_tmdb(row["title"])
        rec_match2 = next((r for r in rec_results2 if r.get("media_type") in ["movie","tv"]), None)
        if rec_match2:
            rec_id2 = rec_match2["id"]
            rec_type2 = rec_match2["media_type"]
            rec_link2 = f"https://www.themoviedb.org/{rec_type2}/{rec_id2}"
            rec_trailer2 = get_trailer(rec_id2, rec_type2)
            rec_providers2 = get_watch_providers(rec_id2, rec_type2)
            stream2 = rec_providers2.get("flatrate", [])
            rec_stream2 = ", ".join([p["provider_name"] for p in stream2[:3]]) if stream2 else None
            st.markdown(f"**[{row['title']}]({rec_link2})** ({row['year']})")
            st.caption(f"{row['type']} | ⭐ {row['rating']:.1f}")
            if rec_trailer2:
                st.markdown(f"[▶ Trailer]({rec_trailer2})")
            if rec_stream2:
                st.caption(f"Stream: {rec_stream2}")
        else:
            st.markdown(f"**{row['title']}** ({row['year']})")
            st.caption(f"{row['type']} | ⭐ {row['rating']:.1f}")

st.markdown("---")
st.caption("Powered by TMDB API | TF-IDF + Cosine Similarity | Watch providers by JustWatch via TMDB")