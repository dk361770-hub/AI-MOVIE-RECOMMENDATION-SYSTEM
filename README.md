# ReelSense — AI Movie Recommendation System

ReelSense is a small, local movie-discovery app. Get recommendations from a movie
you enjoyed, or combine genres, a mood, and favorite movies to describe what
you want to watch.

## Features

- Content-based recommendations using TF-IDF and cosine similarity.
- Preference recommendations from genres, moods, and multiple favorite movies.
- A bundled 36-movie catalog; no API key, account, or network request is needed.
- Match scores show relative similarity, not a prediction or probability.

## Run on Windows

From the project folder in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser. If PowerShell blocks virtual-environment
activation, use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Run tests

```powershell
python -m pytest
```

## How it works

Each movie's genres, mood, keywords, and synopsis are combined into a text
document. Scikit-learn converts those documents into TF-IDF vectors, and cosine
similarity ranks movies with related content. Preference mode can also apply
genre overlap and similarities to selected favorites. The starter catalog lives
in `data/movies.csv` and can be extended with additional rows using the same
columns.
