"""Streamlit interface for the AI Movie Recommendation System."""

from pathlib import Path

import streamlit as st

from recommender import MovieRecommender, load_movies


st.set_page_config(
    page_title="ReelSense — Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "data" / "movies.csv"


@st.cache_data
def get_catalog():
    return load_movies(str(DATA_PATH))


@st.cache_resource
def get_recommender():
    return MovieRecommender(get_catalog())


st.markdown(
    """
    <style>
    .stApp { background: #0c0d12; }
    [data-testid="stSidebar"] { background: #13151d; }
    .hero {
        padding: 2rem 2.2rem;
        border: 1px solid #292c38;
        border-radius: 22px;
        background: radial-gradient(ellipse at 85% 15%, #392450 0, #191722 42%, #13151d 100%);
        margin: 0.5rem 0 1.7rem;
    }
    .hero h1 { font-size: clamp(2.2rem, 5vw, 3.5rem); margin: 0; letter-spacing: -0.05em; }
    .hero p { color: #b7b7c5; font-size: 1.05rem; margin: 0.7rem 0 0; }
    .eyebrow { color: #c9a7ff; font-weight: 700; text-transform: uppercase; letter-spacing: 0.14em; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #151721; border-color: #292c38; border-radius: 16px;
    }
    .movie-meta { color: #a4a6b5; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

movies = get_catalog()
recommender = get_recommender()
all_genres = sorted(
    {genre.strip() for value in movies["genres"] for genre in value.split("|") if genre.strip()}
)
all_moods = sorted(
    {mood.strip() for value in movies["mood"] for mood in value.split("|") if mood.strip()}
)
movie_labels = {
    row["title"]: f'{row["title"]} ({row["year"]})' for _, row in movies.iterrows()
}

with st.sidebar:
    st.markdown("## 🎞️ ReelSense")
    st.caption("A little inspiration for your next movie night.")
    mode = st.radio(
        "How should we find a movie?",
        ["Because I liked a movie", "Pick my vibe"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("### Your movie night")
    selected_genres = st.multiselect("Genres", all_genres, placeholder="Choose one or more")
    selected_mood = st.selectbox("Mood", ["Any mood", *all_moods])
    limit = st.slider("How many suggestions?", min_value=3, max_value=10, value=6)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Your next favorite is waiting</div>
      <h1>Find your next <span style="color:#c9a7ff">great watch.</span></h1>
      <p>Tell us what you love or choose a vibe. We'll find movies with something in common.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.2, 0.8], gap="large")
with left:
    if mode == "Because I liked a movie":
        chosen_title = st.selectbox(
            "Pick a movie you enjoyed",
            movies["title"].tolist(),
            format_func=lambda title: movie_labels[title],
        )
        selected = movies.loc[movies["title"] == chosen_title].iloc[0]
        st.markdown(f"### {selected['title']} <span class='movie-meta'>({selected['year']})</span>", unsafe_allow_html=True)
        st.write(selected["overview"])
        st.caption(f"**{selected['genres'].replace('|', ' · ')}**  ·  ⭐ {selected['rating']}/10")
        recommendations = recommender.recommend_similar(chosen_title, limit)
    else:
        favorite_titles = st.multiselect(
            "Optionally, add movies you already love",
            movies["title"].tolist(),
            format_func=lambda title: movie_labels[title],
            placeholder="Search and select favorites",
        )
        effective_mood = "" if selected_mood == "Any mood" else selected_mood
        if not selected_genres and not effective_mood and not favorite_titles:
            st.info("Choose at least one genre, mood, or favorite movie in the sidebar.")
            recommendations = None
        else:
            recommendations = recommender.recommend_for_preferences(
                genres=selected_genres,
                mood=effective_mood,
                favorite_titles=favorite_titles,
                limit=limit,
            )

with right:
    st.markdown("### The ReelSense approach")
    st.write(
        "Recommendations are based on movie descriptions, genres, themes, and mood. "
        "The more your picks have in common, the higher they rank."
    )
    metric_a, metric_b = st.columns(2)
    metric_a.metric("Movies to explore", len(movies))
    metric_b.metric("Genres", len(all_genres))
    st.caption("No account, tracking, or external movie API needed.")

st.divider()
if recommendations is not None:
    st.markdown("## Picks for you")
    if recommendations.empty:
        st.warning("No matching movies found. Try a different movie or preference.")
    else:
        cards = st.columns(3, gap="medium")
        for index, (_, movie) in enumerate(recommendations.iterrows()):
            with cards[index % 3]:
                with st.container(border=True):
                    st.markdown(f"### {movie['title']}")
                    st.caption(
                        f"{movie['year']}  ·  ⭐ {movie['rating']}/10  ·  "
                        f"{movie['genres'].replace('|', ' / ')}"
                    )
                    st.write(movie["overview"])
                    st.progress(float(movie["match_score"]), text="Taste match")
    with st.expander("How are these recommendations calculated?"):
        st.write(
            "ReelSense converts each movie's genres, mood, keywords, and synopsis into "
            "TF-IDF vectors, then compares them with cosine similarity. Genre overlap "
            "and your selected preferences help refine the ranking. The match bar is a "
            "relative similarity score, not a probability."
        )
