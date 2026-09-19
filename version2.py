import base64
import json
import os
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st


from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


health = pd.read_csv("therapy.csv")
spotify = pd.read_csv("spotify.csv")

print(health.head())
print(spotify.head())

print("Health columns:")
print(health.columns.tolist())

print("Spotify columns:")
print(spotify.columns.tolist())

# ---- Health dataset columns ----
PRE_STRESS_COLUMN = "Pre_session_label"
POST_STRESS_COLUMN = "Post_session_label"
HEALTH_VALENCE_COLUMN = "Valence_spectral"
HEALTH_AROUSAL_COLUMN = "Arousal_spectral"

# ---- Spotify dataset columns ----
TRACK_NAME_COLUMN = "track_name"
ARTISTS_COLUMN = "artists"
SPOTIFY_VALENCE_COLUMN = "valence"
ENERGY_COLUMN = "energy"

if "track_genre" in spotify.columns:
    GENRE_COLUMN = "track_genre"
else:
    GENRE_COLUMN = None


# ---- Build health_model ----
health_model = health[
    [PRE_STRESS_COLUMN, POST_STRESS_COLUMN, HEALTH_VALENCE_COLUMN, HEALTH_AROUSAL_COLUMN]
].copy()

print(health_model.head())

# ---- Build spotify_model ----
spotify_model = spotify[
    [TRACK_NAME_COLUMN, ARTISTS_COLUMN, SPOTIFY_VALENCE_COLUMN, ENERGY_COLUMN]
].copy()

print(spotify_model.head())

# These ARE numeric — safe to coerce
health_model[HEALTH_VALENCE_COLUMN] = pd.to_numeric(
    health_model[HEALTH_VALENCE_COLUMN], errors="coerce"
)
health_model[HEALTH_AROUSAL_COLUMN] = pd.to_numeric(
    health_model[HEALTH_AROUSAL_COLUMN], errors="coerce"
)

# Pre_session_label IS ordinal stress -> preserve low->high order
stress_order = {"low_stress": 0, "moderate_stress": 1, "high_stress": 2}
health_model[PRE_STRESS_COLUMN] = health_model[PRE_STRESS_COLUMN].map(stress_order)

# Post_session_label is just an outcome category -> LabelEncoder is fine
le = LabelEncoder()
health_model[POST_STRESS_COLUMN] = le.fit_transform(health_model[POST_STRESS_COLUMN].astype(str))

print(health_model.head())
print(health_model.isna().sum())  # should now show 0 across the board

stress_scaler = MinMaxScaler()
valence_scaler = MinMaxScaler()
arousal_scaler = MinMaxScaler()

health_model[[PRE_STRESS_COLUMN]] = stress_scaler.fit_transform(health_model[[PRE_STRESS_COLUMN]])
health_model[[HEALTH_VALENCE_COLUMN]] = valence_scaler.fit_transform(health_model[[HEALTH_VALENCE_COLUMN]])
health_model[[HEALTH_AROUSAL_COLUMN]] = arousal_scaler.fit_transform(health_model[[HEALTH_AROUSAL_COLUMN]])

print(health_model.head())


X = health_model[[PRE_STRESS_COLUMN]]

y = health_model[
    [
        HEALTH_VALENCE_COLUMN,
        HEALTH_AROUSAL_COLUMN
    ]
]

print(X.head())
print(y.head())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training:", X_train.shape)
print("Testing:", X_test.shape)

base_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model = MultiOutputRegressor(
    base_model
)

model.fit(
    X_train,
    y_train
)

print("Model trained")

predictions = model.predict(X_test)
predicted_valence = predictions[:, 0]
predicted_arousal = predictions[:, 1]
print(predictions[:10])

actual_valence = y_test[HEALTH_VALENCE_COLUMN]
actual_arousal = y_test[HEALTH_AROUSAL_COLUMN]

valence_error = mean_absolute_error(
    actual_valence,
    predicted_valence
)

arousal_error = mean_absolute_error(
    actual_arousal,
    predicted_arousal
)

print("Valence MAE:", valence_error)
print("Arousal MAE:", arousal_error)




# ---- Build spotify_model ----
spotify_model = spotify.copy()

spotify_model[SPOTIFY_VALENCE_COLUMN] = pd.to_numeric(
    spotify_model[SPOTIFY_VALENCE_COLUMN],
    errors="coerce"
)

spotify_model[ENERGY_COLUMN] = pd.to_numeric(
    spotify_model[ENERGY_COLUMN],
    errors="coerce"
)

spotify_model = spotify_model.dropna(
    subset=[
        SPOTIFY_VALENCE_COLUMN,
        ENERGY_COLUMN
    ]
)

print(
    spotify_model[
        [
            TRACK_NAME_COLUMN,
            ARTISTS_COLUMN,
            SPOTIFY_VALENCE_COLUMN,
            ENERGY_COLUMN
        ]
    ].head()
)

spotify_model = spotify_model[
    spotify_model[SPOTIFY_VALENCE_COLUMN].between(0, 1)
    &
    spotify_model[ENERGY_COLUMN].between(0, 1)
]

print("Number of songs:", len(spotify_model))

print(
    spotify_model[
        [
            SPOTIFY_VALENCE_COLUMN,
            ENERGY_COLUMN
        ]
    ].describe()
)


def predict_emotional_state(stress_value):

    stress_df = pd.DataFrame(
        {
            PRE_STRESS_COLUMN: [stress_value]
        }
    )

    normalized_stress = stress_scaler.transform(
        stress_df
    )[0][0]

    normalized_stress = np.clip(
        normalized_stress,
        0,
        1
    )

    model_input = pd.DataFrame(
        {
            PRE_STRESS_COLUMN: [
                normalized_stress
            ]
        }
    )

    prediction = model.predict(
        model_input
    )[0]

    current_valence = prediction[0]
    current_arousal = prediction[1]

    return (
        normalized_stress,
        current_valence,
        current_arousal
    )


# ============================================
# CALCULATE TARGET EMOTIONAL STATE
# ============================================

def calculate_target_state(stress_value):
    (
        normalized_stress,
        current_valence,
        current_arousal
    ) = predict_emotional_state(stress_value)

    # Increase valence more when stress is higher
    valence_increase = (
            0.10
            +
            0.20 * normalized_stress
    )

    target_valence = (
            current_valence
            +

            valence_increase
    )

    target_valence = np.clip(
        target_valence,
        0,
        1
    )

    # Move arousal toward a calmer level
    calm_arousal = 0.40

    movement_strength = (
            0.30
            +
            0.50 * normalized_stress
    )

    target_arousal = (
            current_arousal
            +
            movement_strength
            *
            (
                calm_arousal
                -
                current_arousal
            )
    )

    target_arousal = np.clip(
        target_arousal,
        0,
        1
    )

    return {
        "normalized_stress": normalized_stress,
        "current_valence": current_valence,
        "current_arousal": current_arousal,
        "target_valence": target_valence,
        "target_arousal": target_arousal
    }


# ============================================
# EMOTIONAL MUSIC MATCHING
# ============================================

def calculate_emotional_matches(
        target_valence,
        target_arousal
):
    songs = spotify_model.copy()

    # Difference between song valence
    # and target valence
    songs["valence_distance"] = abs(
        songs[SPOTIFY_VALENCE_COLUMN]
        -
        target_valence
    )

    # Spotify energy is being used
    # as the music-side approximation of arousal
    songs["arousal_distance"] = abs(
        songs[ENERGY_COLUMN]
        -
        target_arousal
    )

    VALENCE_WEIGHT = 0.40
    AROUSAL_WEIGHT = 0.60

    songs["emotional_distance"] = np.sqrt(

        VALENCE_WEIGHT
        *
        (
                songs["valence_distance"] ** 2
        )

        +

        AROUSAL_WEIGHT
        *
        (
                songs["arousal_distance"] ** 2
        )
    )

    return songs





# Fallback credentials for local runs. Prefer Streamlit secrets on Cloud.
DEFAULT_CLIENT_ID = "0b585bb6836640b19329bb66d05f2acc"
DEFAULT_CLIENT_SECRET = "e881ea0764664d9484f3fe4ce463f043"
LOCAL_REDIRECT_URI = "http://127.0.0.1:8501/"
SCOPES = (
    "user-read-private "
    "user-read-email "
    "user-top-read"
)

TOKEN_FILE = ".spotify_token.json"

AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
PROFILE_URL = "https://api.spotify.com/v1/me"


def is_streamlit_cloud():

    return (
        Path("/mount/src").exists()
        or os.environ.get("STREAMLIT_RUNTIME_ENV") == "cloud"
    )


def get_secret(name, default=None):

    try:
        return st.secrets[name]
    except Exception:
        return default


def get_spotify_client_id():

    return get_secret(
        "SPOTIFY_CLIENT_ID",
        DEFAULT_CLIENT_ID
    )


def get_spotify_client_secret():

    return get_secret(
        "SPOTIFY_CLIENT_SECRET",
        DEFAULT_CLIENT_SECRET
    )


def get_redirect_uri():

    configured = get_secret("SPOTIFY_REDIRECT_URI")

    if configured:

        return configured.rstrip("/") + "/"

    if is_streamlit_cloud():

        try:

            host = (
                st.context.headers.get("Host")
                or st.context.headers.get("host")
            )

        except Exception:

            host = None

        if host:

            return f"https://{host}/"

        raise RuntimeError(
            "Set SPOTIFY_REDIRECT_URI in Streamlit secrets to your "
            "exact app URL, for example "
            "https://your-app.streamlit.app/"
        )

    return LOCAL_REDIRECT_URI


def build_spotify_auth_url():

    auth_params = urllib.parse.urlencode(
        {
            "client_id": get_spotify_client_id(),
            "response_type": "code",
            "redirect_uri": get_redirect_uri(),
            "scope": SCOPES,
            "show_dialog": "true",
        }
    )

    return f"{AUTH_URL}?{auth_params}"


def exchange_code_for_token(code):

    client_id = get_spotify_client_id()
    client_secret = get_spotify_client_secret()
    auth_string = f"{client_id}:{client_secret}"
    auth_base64 = base64.b64encode(
        auth_string.encode("utf-8")
    ).decode("utf-8")

    response = requests.post(
        TOKEN_URL,
        headers={
            "Authorization": f"Basic {auth_base64}"
        },
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": get_redirect_uri(),
        },
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Token exchange failed ({response.status_code}): {response.text}"
        )

    return response.json()


def refresh_access_token(refresh_token):

    client_id = get_spotify_client_id()
    client_secret = get_spotify_client_secret()
    auth_string = f"{client_id}:{client_secret}"
    auth_base64 = base64.b64encode(
        auth_string.encode("utf-8")
    ).decode("utf-8")

    response = requests.post(
        TOKEN_URL,
        headers={
            "Authorization": f"Basic {auth_base64}"
        },
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Token refresh failed ({response.status_code}): {response.text}"
        )

    return response.json()


def load_saved_token():

    saved = st.session_state.get("spotify_token_data")

    if saved:

        return saved

    # Shared token files are unsafe for multi-user Cloud apps
    if is_streamlit_cloud():

        return None

    try:

        with open(TOKEN_FILE, encoding="utf-8") as file:

            return json.load(file)

    except (FileNotFoundError, json.JSONDecodeError):

        return None


def save_token(token_data):

    st.session_state["spotify_token_data"] = token_data

    if is_streamlit_cloud():

        return

    with open(TOKEN_FILE, "w", encoding="utf-8") as file:

        json.dump(token_data, file, indent=2)


def has_required_scopes(token_data):

    granted = set(
        token_data.get("scope", "").split()
    )

    return set(SCOPES.split()).issubset(granted)


def refresh_saved_access_token():

    saved = load_saved_token()

    if not saved or not saved.get("refresh_token"):

        return None

    # Refreshing keeps the scopes of the original login, so an older
    # token has to be replaced rather than reused
    if not has_required_scopes(saved):

        return None

    try:

        refreshed = refresh_access_token(
            saved["refresh_token"]
        )

    except RuntimeError:

        return None

    # Spotify only returns a new refresh token some of the time
    refresh_token = saved["refresh_token"]
    saved.update(refreshed)
    saved["refresh_token"] = refresh_token
    save_token(saved)

    return saved["access_token"]


def handle_spotify_callback():

    query_params = st.query_params

    if "error" in query_params:

        error = query_params.get("error")
        st.query_params.clear()
        st.error(
            f"Spotify login was denied or failed: {error}"
        )
        return False

    if "code" not in query_params:

        return False

    code = query_params.get("code")

    if isinstance(code, list):

        code = code[0]

    try:

        token_data = exchange_code_for_token(code)
        save_token(token_data)
        connected = connect_spotify(
            token_data["access_token"]
        )

    except RuntimeError as error:

        st.query_params.clear()
        st.error(
            f"Spotify login failed: {error}"
        )
        return False

    st.query_params.clear()

    return connected


# ============================================
# SPOTIFY USER PREFERENCES
# ============================================

def get_user_top_tracks(
    access_token,
    limit=30
):

    url = (
        "https://api.spotify.com/v1/"
        "me/top/tracks"
    )

    headers = {
        "Authorization":
            f"Bearer {access_token}"
    }

    params = {
        "limit": limit,
        "time_range": "medium_term"
    }

    response = requests.get(
        url,
        headers=headers,
        params=params
    )

    if response.status_code != 200:

        raise RuntimeError(
            "Could not retrieve Spotify preferences "
            f"({response.status_code}): {response.text}"
        )

    return response.json().get(
        "items",
        []
    )

def get_artist_genres(
    access_token,
    artist_ids
):

    genres = set()

    unique_ids = list(
        dict.fromkeys(artist_ids)
    )

    # The artists endpoint accepts 50 ids per request
    for start in range(0, len(unique_ids), 50):

        batch = unique_ids[start:start + 50]

        response = requests.get(
            "https://api.spotify.com/v1/artists",
            headers={
                "Authorization":
                    f"Bearer {access_token}"
            },
            params={
                "ids": ",".join(batch)
            }
        )

        if response.status_code != 200:
            continue

        for artist in response.json().get(
            "artists",
            []
        ):

            if not artist:
                continue

            for genre in artist.get(
                "genres",
                []
            ):

                genres.add(
                    genre.strip().lower()
                )

    return genres


# ============================================
# PERSONAL MUSIC PREFERENCES FROM SPOTIFY
# ============================================

def extract_artist_ids(
    top_tracks
):

    artist_ids = []

    for track in top_tracks:

        for artist in track.get(
            "artists",
            []
        ):

            artist_id = artist.get("id")

            if artist_id:

                artist_ids.append(artist_id)

    return artist_ids


def extract_spotify_preferences(
    top_tracks
):

    favorite_artists = []

    favorite_songs = []

    for track in top_tracks:

        track_name = (
            track.get(
                "name",
                ""
            )
            .strip()
            .lower()
        )

        if track_name:

            favorite_songs.append(
                track_name
            )

        for artist in track.get(
            "artists",
            []
        ):

            artist_name = (
                artist.get(
                    "name",
                    ""
                )
                .strip()
                .lower()
            )

            if artist_name:

                favorite_artists.append(
                    artist_name
                )

    favorite_artists = list(
        set(
            favorite_artists
        )
    )

    favorite_songs = list(
        set(
            favorite_songs
        )
    )

    return (
        favorite_artists,
        favorite_songs
    )


# ============================================
# SEASONAL FILTER
# ============================================

HOLIDAY_MONTH = 12

# Matched against track titles. Terms like "santa" and "merry" on their
# own are left out because they also appear in "Santa Monica" and
# "Merry-Go-Round of Life"
HOLIDAY_PATTERN = "|".join([
    r"\bchristmas\b",
    r"\bx-?mas\b",
    r"\bsanta claus\b",
    r"\bsanta baby\b",
    r"\bjingle bells?\b",
    r"\bnavidad\b",
    r"\bno\u00ebl\b",
    r"\bnoel\b",
    r"\bsleigh\b",
    r"\brudolph\b",
    r"\bsilent night\b",
    r"\bmistletoe\b",
    r"\bwinter wonderland\b",
    r"\bauld lang syne\b",
    r"\bhanukk?ah\b",
    r"\bdreidel\b",
    r"\blittle drummer boy\b",
    r"\bdeck the halls?\b",
    r"\blet it snow\b",
    r"\bfrosty the snowman\b",
    r"\bo holy night\b",
    r"\bsilver bells\b",
    r"\bweihnacht",
    r"\byule\b",
    r"\bholly jolly\b",
    r"\bwenceslas\b",
    r"\breindeer\b",
    r"\bst\.? nick\b"
])


def is_holiday_music(songs):

    return (
        songs[TRACK_NAME_COLUMN]
        .astype(str)
        .str.lower()
        .str.contains(
            HOLIDAY_PATTERN,
            regex=True,
            na=False
        )
    )


def remove_out_of_season_music(
    songs,
    today=None
):

    if today is None:

        today = pd.Timestamp.now()

    if today.month == HOLIDAY_MONTH:

        return songs

    return songs[
        ~is_holiday_music(songs)
    ]


# ============================================
# MATCH SPOTIFY GENRES TO THE CATALOG
# ============================================

# Spotify uses free-form artist genres, the catalog uses 114 fixed labels,
# so a few common spellings need to be bridged by hand
GENRE_ALIASES = {
    "r-n-b": {"rnb", "rb", "randb"},
    "hip-hop": {"hiphop"},
    "drum-and-bass": {"drumandbass", "dnb"},
    "rock-n-roll": {"rocknroll", "rockandroll"},
    "world-music": {"world"},
    "singer-songwriter": {"singersongwriter"},
    "trip-hop": {"triphop"},
    "synth-pop": {"synthpop"},
    "indie-pop": {"indiepop"},
    "power-pop": {"powerpop"},
    "punk-rock": {"punkrock"},
    "hard-rock": {"hardrock"},
    "heavy-metal": {"heavymetal"},
    "new-age": {"newage"},
    "alt-rock": {"altrock", "alternativerock"}
}


def normalize_genre_text(text):

    return "".join(
        character
        for character in text.lower()
        if character.isalnum()
    )


def match_catalog_genres(spotify_genres):

    if GENRE_COLUMN is None:

        return set()

    # Match on the whole genre and on each word inside it, so that
    # "new wave pop" can still line up with the catalog's "pop"
    listened = set()

    for genre in spotify_genres:

        listened.add(
            normalize_genre_text(genre)
        )

        for word in genre.split():

            listened.add(
                normalize_genre_text(word)
            )

    catalog_genres = set(
        spotify_model[GENRE_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    matched = set()

    for catalog_genre in catalog_genres:

        spellings = {
            normalize_genre_text(catalog_genre)
        }

        spellings |= GENRE_ALIASES.get(
            catalog_genre,
            set()
        )

        if spellings & listened:

            matched.add(catalog_genre)

    return matched


# ============================================
# SCORE SONGS AGAINST SPOTIFY PREFERENCES
# ============================================

def add_preference_scores(
    songs,
    favorite_artists,
    favorite_songs,
    favorite_genres
):

    personalized = songs.copy()

    favorite_artists = set(favorite_artists)

    favorite_songs = set(favorite_songs)

    favorite_genres = set(favorite_genres)

    track_names = (
        personalized[TRACK_NAME_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    artist_names = (
        personalized[ARTISTS_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    title_match = track_names.isin(
        favorite_songs
    )

    def has_favorite_artist(value):

        # A track can credit several artists, as in "artist a;artist b"
        for artist in value.split(";"):

            if artist.strip() in favorite_artists:

                return True

        return False

    artist_match = artist_names.apply(
        has_favorite_artist
    )

    # Titles repeat across the catalog, so a track only counts as one
    # the user listens to when the artist lines up as well
    song_match = title_match & artist_match

    if GENRE_COLUMN is not None:

        genre_match = (
            personalized[GENRE_COLUMN]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(favorite_genres)
        )

    else:

        genre_match = pd.Series(
            False,
            index=personalized.index
        )

    SONG_MATCH_SCORE = 1.00
    ARTIST_MATCH_SCORE = 0.70
    GENRE_MATCH_SCORE = 0.35

    personalized["preference_score"] = np.where(

        song_match,

        SONG_MATCH_SCORE,

        np.where(

            artist_match,

            ARTIST_MATCH_SCORE,

            np.where(
                genre_match,
                GENRE_MATCH_SCORE,
                0.0
            )
        )
    )

    return personalized


# ============================================
# FEEDBACK MEMORY
# ============================================

FEEDBACK_FILE = "recommendation_feedback.csv"


def load_feedback():

    if Path(FEEDBACK_FILE).exists():

        feedback = pd.read_csv(
            FEEDBACK_FILE
        )

    else:

        feedback = pd.DataFrame(
            columns=[
                "user_id",
                "track_name",
                "artists",
                "pre_stress",
                "post_stress",
                "helpfulness",
                "timestamp"
            ]
        )

    return feedback


# ============================================
# SPOTIFY USER PROFILE
# ============================================

def get_spotify_profile(access_token):

    response = requests.get(
        PROFILE_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
    )

    if response.status_code != 200:
        raise RuntimeError(
            "Could not retrieve Spotify profile."
        )

    return response.json()


def connect_spotify(access_token):

    try:
        profile = get_spotify_profile(access_token)
    except RuntimeError:
        return False

    st.session_state["spotify_access_token"] = access_token
    st.session_state["spotify_profile"] = profile

    return True


def restore_spotify_session():

    access_token = refresh_saved_access_token()

    if access_token is None:
        return False

    return connect_spotify(access_token)


def forget_spotify_session():

    if not is_streamlit_cloud():

        Path(TOKEN_FILE).unlink(
            missing_ok=True
        )

    for key in [
        "spotify_access_token",
        "spotify_profile",
        "spotify_token_data",
        "recommendations",
        "emotional_state",
        "taste"
    ]:

        st.session_state.pop(
            key,
            None
        )


# ============================================
# LEARN FROM PREVIOUS FEEDBACK
# ============================================

def add_feedback_scores(
    songs,
    user_id,
    stress_level
):

    personalized = songs.copy()

    feedback = load_feedback()

    # Unique song identifier
    personalized["song_key"] = (

        personalized[TRACK_NAME_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()

        +

        " | "

        +

        personalized[ARTISTS_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Artist identifier
    personalized["artist_key"] = (

        personalized[ARTISTS_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # No previous feedback yet
    if feedback.empty:

        personalized["feedback_score"] = 0.0

        return personalized

    # Only use this user's previous results
    feedback = feedback[
        (
            feedback["user_id"] == user_id
        )
        &
        (
            feedback["pre_stress"] == stress_level
        )
    ].copy()

    if feedback.empty:

        personalized["feedback_score"] = 0.0

        return personalized

    feedback["song_key"] = (

        feedback["track_name"]
        .astype(str)
        .str.strip()
        .str.lower()

        +

        " | "

        +

        feedback["artists"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    feedback["artist_key"] = (

        feedback["artists"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Turn helpfulness rating into
    # values between -1 and +1
    feedback["helpfulness_score"] = (

        feedback["helpfulness"]
        -
        3

    ) / 2

    stress_numbers = {
        "Low Stress": 0,
        "Moderate Stress": 1,
        "High Stress": 2
    }

    feedback["pre_number"] = (
        feedback["pre_stress"]
        .map(stress_numbers)
    )

    feedback["post_number"] = (
        feedback["post_stress"]
        .map(stress_numbers)
    )

    # Positive number means stress decreased
    feedback["stress_improvement"] = (

        feedback["pre_number"]
        -
        feedback["post_number"]

    ) / 2

    # Combine subjective helpfulness
    # and reported stress change
    feedback["feedback_effect"] = (

        0.70
        *
        feedback["helpfulness_score"]

        +

        0.30
        *
        feedback["stress_improvement"]
    )

    # Average previous result for each song
    song_history = (

        feedback
        .groupby("song_key")[
            "feedback_effect"
        ]
        .mean()
    )

    # Average previous result for each artist
    artist_history = (

        feedback
        .groupby("artist_key")[
            "feedback_effect"
        ]
        .mean()
    )

    personalized["song_feedback"] = (

        personalized["song_key"]
        .map(song_history)
        .fillna(0)
    )

    personalized["artist_feedback"] = (

        personalized["artist_key"]
        .map(artist_history)
        .fillna(0)
    )

    personalized["feedback_score"] = (

        0.65
        *
        personalized["song_feedback"]

        +

        0.35
        *
        personalized["artist_feedback"]
    )

    return personalized



# ============================================
# FINAL PERSONALIZED SCORE
# ============================================

def calculate_final_score(songs):

    personalized = songs.copy()

    PREFERENCE_BONUS = 0.25
    FEEDBACK_BONUS = 0.10

    personalized["final_score"] = (

        personalized["emotional_distance"]

        -

        PREFERENCE_BONUS
        *
        personalized["preference_score"]

        -

        FEEDBACK_BONUS
        *
        personalized["feedback_score"]
    )

    # Lower score = better recommendation
    personalized = personalized.sort_values(
        "final_score",
        ascending=True
    )

    # Avoid duplicate versions of the same track
    personalized = personalized.drop_duplicates(
        subset=[
            TRACK_NAME_COLUMN,
            ARTISTS_COLUMN
        ]
    )

    return personalized


# ============================================
# COMPLETE VERSION 2 PIPELINE
# ============================================

def recommend_songs(
    stress_value,
    stress_level,
    user_id,
    spotify_top_tracks,
    access_token,
    number_of_songs=10
):

    # 1. Predict emotional state
    emotional_state = (
        calculate_target_state(
            stress_value
        )
    )

    # 2. Find emotional matches
    songs = (
        calculate_emotional_matches(

            emotional_state[
                "target_valence"
            ],

            emotional_state[
                "target_arousal"
            ]
        )
    )

    # 3. Hold back holiday music outside December
    songs = remove_out_of_season_music(
        songs
    )

    # 4. Convert Spotify listening
    # history into preferences
    (
        favorite_artists,
        favorite_songs
    ) = extract_spotify_preferences(
        spotify_top_tracks
    )

    spotify_genres = get_artist_genres(

        access_token,

        extract_artist_ids(
            spotify_top_tracks
        )
    )

    favorite_genres = match_catalog_genres(
        spotify_genres
    )

    # 5. Add Spotify preference score
    songs = add_preference_scores(

        songs,

        favorite_artists,

        favorite_songs,

        favorite_genres
    )

    # 6. Keep only music that matches the
    # user's taste, as long as that leaves
    # enough candidates to choose from
    familiar = songs[
        songs["preference_score"] > 0
    ]

    if len(familiar) >= number_of_songs:

        songs = familiar.copy()

    # 7. Add previous feedback
    songs = add_feedback_scores(

        songs,

        user_id,

        stress_level
    )

    # 8. Calculate final score
    songs = calculate_final_score(
        songs
    )

    recommendations = (
        songs
        .head(
            number_of_songs
        )
        .copy()
    )

    taste = {
        "favorite_genres": sorted(
            favorite_genres
        ),
        "favorite_artists": favorite_artists,
        "candidate_count": len(songs)
    }

    return (
        emotional_state,
        recommendations,
        taste
    )


# ============================================
# SAVE USER FEEDBACK
# ============================================

def save_feedback(
    user_id,
    track_name,
    artists,
    pre_stress,
    post_stress,
    helpfulness
):

    feedback = load_feedback()

    new_feedback = pd.DataFrame(
        [
            {
                "user_id": user_id,
                "track_name": track_name,
                "artists": artists,
                "pre_stress": pre_stress,
                "post_stress": post_stress,
                "helpfulness": helpfulness,
                "timestamp": pd.Timestamp.now()
            }
        ]
    )

    feedback = pd.concat(
        [
            feedback,
            new_feedback
        ],
        ignore_index=True
    )

    feedback.to_csv(
        FEEDBACK_FILE,
        index=False
    )

st.title(
    "Stress-Aware Music Recommender"
)

st.write(
    """
    This system combines stress-related
    health data, musical characteristics,
    Spotify listening behavior, and user
    feedback to create personalized music
    recommendations.
    """
)
# ============================================
# SPOTIFY LOGIN
# ============================================

# Spotify redirects back here with ?code=... after the user approves access
if "spotify_access_token" not in st.session_state:

    if handle_spotify_callback():

        st.rerun()

# Returning users are reconnected from a saved refresh token
if "spotify_access_token" not in st.session_state:

    restore_spotify_session()

if "spotify_access_token" not in st.session_state:

    st.subheader(
        "Step 1: Connect your Spotify account"
    )

    st.write(
        """
        Recommendations are based on your Spotify
        listening history, so you need to log in
        before you can continue.
        """
    )

    try:

        redirect_uri = get_redirect_uri()
        auth_url = build_spotify_auth_url()

    except RuntimeError as error:

        st.error(str(error))
        st.stop()

    st.info(
        "In your Spotify Developer Dashboard, add this "
        "exact Redirect URI and click Save:\n\n"
        f"`{redirect_uri}`"
    )

    st.link_button(
        "Log in with Spotify",
        auth_url,
        type="primary"
    )

    # Nothing below this point runs until Spotify is connected
    st.stop()


profile = (
    st.session_state[
        "spotify_profile"
    ]
)

st.success(
    "Spotify connected: "
    +
    profile.get(
        "display_name",
        "Spotify User"
    )
)

if st.button(
    "Disconnect Spotify"
):

    forget_spotify_session()

    st.rerun()

stress_level = st.selectbox(
    "How stressed are you right now?",
    [
        "Low Stress",
        "Moderate Stress",
        "High Stress"
    ]
)

stress_map = {
    "Low Stress": 0,
    "Moderate Stress": 1,
    "High Stress": 2
}

stress_value = stress_map[
    stress_level
]


if st.button(
    "Recommend Music"
):

    access_token = (
        st.session_state[
            "spotify_access_token"
        ]
    )

    top_tracks = (
        get_user_top_tracks(
            access_token
        )
    )

    user_id = profile["id"]

    with st.spinner(
        "Reading your Spotify taste and "
        "matching it to your stress level..."
    ):

        (
            emotional_state,
            recommendations,
            taste
        ) = recommend_songs(

            stress_value,

            stress_level,

            user_id,

            top_tracks,

            access_token,

            number_of_songs=10
        )

    st.session_state[
        "recommendations"
    ] = recommendations

    st.session_state[
        "emotional_state"
    ] = emotional_state

    st.session_state[
        "taste"
    ] = taste

    st.session_state[
        "user_id"
    ] = user_id

    st.session_state[
        "pre_stress"
    ] = stress_level

# ============================================
# DISPLAY EMOTIONAL STATE
# ============================================

if "emotional_state" in st.session_state:

    emotional_state = (
        st.session_state[
            "emotional_state"
        ]
    )

    st.subheader(
        "Emotional State Analysis"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "Predicted Current State"
        )

        st.metric(
            "Current Valence",
            round(
                emotional_state[
                    "current_valence"
                ],
                3
            )
        )

        st.metric(
            "Current Arousal",
            round(
                emotional_state[
                    "current_arousal"
                ],
                3
            )
        )

    with col2:

        st.write(
            "Target State"
        )

        st.metric(
            "Target Valence",
            round(
                emotional_state[
                    "target_valence"
                ],
                3
            )
        )

        st.metric(
            "Target Arousal",
            round(
                emotional_state[
                    "target_arousal"
                ],
                3
            )
        )


# ============================================
# DISPLAY RECOMMENDATIONS
# ============================================

if "recommendations" in st.session_state:

    recommendations = (
        st.session_state[
            "recommendations"
        ]
    )

    taste = st.session_state.get(
        "taste",
        {}
    )

    st.subheader(
        "Your Top 10 Songs"
    )

    favorite_genres = taste.get(
        "favorite_genres",
        []
    )

    if favorite_genres:

        st.caption(
            "Matched against genres you listen to: "
            +
            ", ".join(favorite_genres)
        )

    else:

        st.caption(
            "No overlap was found between your Spotify genres "
            "and this catalog, so these are ranked on emotional "
            "fit alone."
        )

    if pd.Timestamp.now().month != HOLIDAY_MONTH:

        st.caption(
            "Holiday music is held back until December."
        )

    for rank, (_, song) in enumerate(
        recommendations.iterrows(),
        start=1
    ):

        if GENRE_COLUMN is not None:

            genre_note = f" · {song[GENRE_COLUMN]}"

        else:

            genre_note = ""

        st.markdown(
            f"**{rank}. {song[TRACK_NAME_COLUMN]}** — "
            f"{song[ARTISTS_COLUMN]}{genre_note}"
        )

        st.caption(
            f"valence {song[SPOTIFY_VALENCE_COLUMN]:.2f} · "
            f"energy {song[ENERGY_COLUMN]:.2f} · "
            f"taste match {song['preference_score']:.2f}"
        )

    with st.expander(
        "Show scoring details"
    ):

        display_columns = [
            TRACK_NAME_COLUMN,
            ARTISTS_COLUMN,
            SPOTIFY_VALENCE_COLUMN,
            ENERGY_COLUMN,
            "preference_score",
            "feedback_score",
            "final_score"
        ]

        if GENRE_COLUMN is not None:

            display_columns.insert(
                2,
                GENRE_COLUMN
            )

        st.dataframe(
            recommendations[
                display_columns
            ],
            use_container_width=True,
            hide_index=True
        )


# ============================================
# FEEDBACK ON ONE SONG
# ============================================

if "recommendations" in st.session_state:

    recommendations = (
        st.session_state[
            "recommendations"
        ]
    )

    st.subheader(
        "Rate one of these songs"
    )

    song_labels = [

        f"{rank}. {song[TRACK_NAME_COLUMN]}"
        f" — {song[ARTISTS_COLUMN]}"

        for rank, (_, song) in enumerate(
            recommendations.iterrows(),
            start=1
        )
    ]

    chosen_label = st.selectbox(
        "Which song did you listen to?",
        song_labels
    )

    current_song = recommendations.iloc[
        song_labels.index(chosen_label)
    ]

    helpfulness = st.slider(
        "How much did the song help?",
        min_value=1,
        max_value=5,
        value=3
    )

    post_stress = st.selectbox(
        "How stressed do you feel after listening?",
        [
            "Low Stress",
            "Moderate Stress",
            "High Stress"
        ]
    )

    if st.button(
        "Save Feedback"
    ):

        save_feedback(

            st.session_state[
                "user_id"
            ],

            current_song[
                TRACK_NAME_COLUMN
            ],

            current_song[
                ARTISTS_COLUMN
            ],

            st.session_state[
                "pre_stress"
            ],

            post_stress,

            helpfulness
        )

        st.success(
            "Feedback saved. "
            "Future recommendations can use this result."
        )


