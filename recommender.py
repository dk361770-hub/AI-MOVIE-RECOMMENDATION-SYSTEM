"""Content-based movie recommendations using TF-IDF and cosine similarity."""

from __future__ import annotations

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


REQUIRED_COLUMNS = {
    "movie_id",
    "title",
    "year",
    "genres",
    "overview",
    "mood",
    "keywords",
    "rating",
}


def _tokens(value: str) -> set[str]:
    return {token.strip().casefold() for token in str(value).split("|") if token.strip()}


class MovieRecommender:
    """Rank movies by similarities in their genres, themes, and descriptions."""

    def __init__(self, movies: pd.DataFrame) -> None:
        missing = REQUIRED_COLUMNS.difference(movies.columns)
        if missing:
            raise ValueError(f"Movie catalog is missing required columns: {', '.join(sorted(missing))}")
        if movies.empty:
            raise ValueError("Movie catalog must contain at least one movie.")
        if movies["title"].duplicated().any():
            raise ValueError("Movie titles must be unique.")

        self.movies = movies.reset_index(drop=True).copy()
        self._title_to_index = {
            str(title).casefold(): index for index, title in enumerate(self.movies["title"])
        }
        documents = self.movies.apply(self._movie_document, axis=1)
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform(documents)
        self._similarities = cosine_similarity(self._matrix)

    @staticmethod
    def _movie_document(movie: pd.Series) -> str:
        return " ".join(
            str(movie[column])
            for column in ("genres", "mood", "keywords", "overview")
            if pd.notna(movie[column])
        )

    def _movie_index(self, title: str) -> int:
        index = self._title_to_index.get(title.casefold())
        if index is None:
            raise ValueError(f"Movie not found in catalog: {title}")
        return index

    def _rank(self, scores: list[float], limit: int, excluded: set[int]) -> pd.DataFrame:
        if limit <= 0:
            return self.movies.iloc[0:0].assign(match_score=pd.Series(dtype=float))

        ranked = sorted(
            (index for index in range(len(scores)) if index not in excluded),
            key=lambda index: (-scores[index], str(self.movies.iloc[index]["title"]).casefold()),
        )[:limit]
        results = self.movies.iloc[ranked].copy()
        results["match_score"] = [round(float(scores[index]), 4) for index in ranked]
        return results.reset_index(drop=True)

    def recommend_similar(self, title: str, limit: int = 6) -> pd.DataFrame:
        """Return movies most similar to a catalog title, excluding that title."""
        source_index = self._movie_index(title)
        scores = self._similarities[source_index].tolist()
        return self._rank(scores, limit, {source_index})

    def recommend_for_preferences(
        self,
        genres: list[str] | None = None,
        mood: str = "",
        favorite_titles: list[str] | None = None,
        limit: int = 6,
    ) -> pd.DataFrame:
        """Recommend from selected genres, a mood, and optionally favorite movies."""
        selected_genres = genres or []
        favorites = favorite_titles or []
        if not selected_genres and not mood.strip() and not favorites:
            raise ValueError("Choose at least one genre, mood, or favorite movie.")

        query = " ".join([*selected_genres, mood.strip()])
        query_vector = self._vectorizer.transform([query])
        preference_scores = cosine_similarity(query_vector, self._matrix).ravel()

        excluded: set[int] = set()
        if favorites:
            favorite_indices = {self._movie_index(title) for title in favorites}
            excluded = favorite_indices
            favorite_scores = self._similarities[list(favorite_indices)].mean(axis=0)
            if query:
                scores = 0.7 * favorite_scores + 0.3 * preference_scores
            else:
                scores = favorite_scores
        else:
            scores = preference_scores

        selected = {genre.casefold() for genre in selected_genres}
        if selected:
            for index, row in self.movies.iterrows():
                movie_genres = _tokens(row["genres"])
                if movie_genres:
                    overlap = len(selected.intersection(movie_genres)) / len(selected.union(movie_genres))
                    scores[index] = 0.75 * scores[index] + 0.25 * overlap

        return self._rank(scores.tolist(), limit, excluded)


def load_movies(path: str) -> pd.DataFrame:
    """Load and normalize the bundled movie catalog."""
    movies = pd.read_csv(path)
    movies["year"] = pd.to_numeric(movies["year"], errors="raise").astype(int)
    movies["rating"] = pd.to_numeric(movies["rating"], errors="raise")
    for column in ("genres", "overview", "mood", "keywords"):
        movies[column] = movies[column].fillna("").astype(str)
    return movies
