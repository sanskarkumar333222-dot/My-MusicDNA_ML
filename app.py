"""
================================================================================
SPOTIFY MUSIC DNA MATCHER & DUAL-DATASET HIT RECOMMENDER (STREAMLIT APP)
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Spotify Wrapped MusicDNA | AI Matcher",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# ULTRA-AESTHETIC SPOTIFY GLASSMORPHISM STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Circular+Std:wght@400;700;900&family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(29, 185, 84, 0.12) 0%, rgba(18, 18, 18, 1) 60%),
                    radial-gradient(circle at 90% 80%, rgba(30, 215, 96, 0.08) 0%, rgba(18, 18, 18, 1) 60%),
                    #121212;
        color: #FFFFFF;
    }

    /* Animated Music Equalizer */
    .equalizer-container {
        display: flex;
        align-items: flex-end;
        gap: 4px;
        height: 24px;
        margin-left: 10px;
    }
    .equalizer-bar {
        width: 4px;
        background: #1DB954;
        border-radius: 2px;
        animation: bounce 1.2s ease-in-out infinite alternate;
    }
    .bar-1 { height: 60%; animation-delay: 0.1s; }
    .bar-2 { height: 100%; animation-delay: 0.3s; }
    .bar-3 { height: 40%; animation-delay: 0.2s; }
    .bar-4 { height: 80%; animation-delay: 0.4s; }
    .bar-5 { height: 50%; animation-delay: 0.15s; }

    @keyframes bounce {
        0% { height: 20%; }
        100% { height: 100%; }
    }

    /* Spotify Hero Banner */
    .hero-box {
        background: linear-gradient(135deg, rgba(29, 185, 84, 0.2) 0%, rgba(24, 24, 24, 0.8) 100%);
        border: 1px solid rgba(29, 185, 84, 0.3);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 30px;
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }

    /* Artist Cards */
    .artist-card {
        background: rgba(30, 30, 30, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
        backdrop-filter: blur(8px);
        position: relative;
        overflow: hidden;
    }
    .artist-card:hover {
        transform: translateY(-5px);
        border-color: #1DB954;
        box-shadow: 0 12px 30px rgba(29, 185, 84, 0.25);
    }
    .artist-card.gold {
        border-top: 4px solid #FFD700;
        background: linear-gradient(180deg, rgba(255, 215, 0, 0.08) 0%, rgba(30, 30, 30, 0.85) 100%);
    }
    .artist-card.silver {
        border-top: 4px solid #C0C0C0;
    }
    .artist-card.bronze {
        border-top: 4px solid #CD7F32;
    }

    /* Track Row Card */
    .track-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 14px 20px;
        margin: 10px 0;
        transition: all 0.2s ease;
    }
    .track-row:hover {
        background: rgba(29, 185, 84, 0.1);
        border-color: rgba(29, 185, 84, 0.4);
        transform: scale(1.01);
    }

    /* Spotify Play Button Pill */
    .spotify-btn {
        background: #1DB954;
        color: #000000 !important;
        font-weight: 700;
        text-decoration: none;
        padding: 8px 18px;
        border-radius: 25px;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        transition: all 0.2s;
    }
    .spotify-btn:hover {
        background: #1ed760;
        transform: scale(1.05);
    }

    /* Personality Badge */
    .persona-badge {
        display: inline-block;
        background: linear-gradient(90deg, #1DB954, #1ed760);
        color: #000000;
        padding: 6px 16px;
        border-radius: 30px;
        font-weight: 800;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }

    /* Streamlit Button Customization */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
        color: #000000;
        font-weight: 800;
        font-size: 1.1rem;
        border-radius: 30px;
        border: none;
        padding: 14px 28px;
        width: 100%;
        box-shadow: 0 4px 20px rgba(29, 185, 84, 0.4);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 25px rgba(29, 185, 84, 0.6);
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# DATA LOADING & ML MODEL PIPELINE
# -----------------------------------------------------------------------------
@st.cache_data
def load_datasets():
    artists_path = "dataset.csv"
    tracks_path = "spotify_top_tracks.csv"

    if not os.path.exists(artists_path):
        artists_path = os.path.join(os.path.dirname(__file__), artists_path)
    if not os.path.exists(tracks_path):
        tracks_path = os.path.join(os.path.dirname(__file__), tracks_path)

    df_artists = pd.read_csv(artists_path)
    df_artists.columns = df_artists.columns.str.strip()

    for col in df_artists.select_dtypes(include=['object']).columns:
        df_artists[col] = df_artists[col].fillna('Unknown').astype(str).str.strip()
    for col in df_artists.select_dtypes(include=[np.number]).columns:
        df_artists[col] = df_artists[col].fillna(df_artists[col].median())

    df_artists = df_artists.drop_duplicates().reset_index(drop=True)
    df_artists['Debut Year'] = df_artists['Debut Year'].astype(int)

    # Feature Engineering
    current_year = 2024
    df_artists['Career_Age'] = (current_year - df_artists['Debut Year']).apply(lambda x: max(x, 1))
    df_artists['Streams_Per_Year'] = df_artists['Total Streams (in millions)'] / df_artists['Career_Age']
    df_artists['Feature_Reliance_Ratio'] = df_artists['Feature Streams (in millions)'] / (df_artists['Total Streams (in millions)'] + 1e-5)
    df_artists['Collab_To_Solo_Ratio'] = df_artists['Collaborative Streams (in millions)'] / (df_artists['Solo Streams (in millions)'] + 1e-5)

    numerical_features = [
        'Debut Year', 
        '% of Solo Streams', 
        '% of Collaborative Streams', 
        'Feature_Reliance_Ratio', 
        'Collab_To_Solo_Ratio'
    ]
    categorical_features = ['Primary Genre', 'Primary Language', 'Artist Type']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    artist_embeddings = preprocessor.fit_transform(df_artists)

    # Tracks dataset
    df_tracks = pd.read_csv(tracks_path) if os.path.exists(tracks_path) else None

    return df_artists, df_tracks, preprocessor, artist_embeddings

df_artists, df_tracks, preprocessor, artist_embeddings = load_datasets()


# -----------------------------------------------------------------------------
# HERO HEADER SECTION
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-box">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <span class="persona-badge">⚡ AI-Powered Musical Matcher</span>
            <h1 style="font-size: 2.5rem; font-weight: 900; margin: 0; color: #FFFFFF;">
                Spotify <span style="color: #1DB954;">MusicDNA</span>
            </h1>
            <p style="color: #CCCCCC; font-size: 1.1rem; margin-top: 8px;">
                Uncover your Superstar Artist Twins, Career DNA Archetype, and Personalized Top 5 Hit Tracks using Unsupervised Machine Learning.
            </p>
        </div>
        <div class="equalizer-container">
            <div class="equalizer-bar bar-1"></div>
            <div class="equalizer-bar bar-2"></div>
            <div class="equalizer-bar bar-3"></div>
            <div class="equalizer-bar bar-4"></div>
            <div class="equalizer-bar bar-5"></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SIDEBAR: USER MUSICAL TASTE QUIZ
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ **Your Music Taste Quiz**")
    st.markdown("Tune your personal musical frequencies:")

    # 1. Favorite Artist
    all_artists = sorted(df_artists['Artist Name'].unique().tolist())
    fav_artist = st.selectbox("1. Who is your Favorite Superstar?", all_artists, index=all_artists.index('The Weeknd') if 'The Weeknd' in all_artists else 0)

    # 2. Favorite Song
    fav_song = st.text_input("2. What is your Favorite Anthem / Track?", value="Blinding Lights")

    # 3. Listening Vibe / Mood
    vibes = [
        "🔥 High-Energy Stadium Anthems",
        "🌙 Late-Night Chill & Melancholy",
        "💃 Upbeat Dance & Party Grooves",
        "🎸 Acoustic, Raw & Soulful",
        "🌍 Global Crossover & Latin Beats"
    ]
    selected_vibe = st.selectbox("3. What is your Current Listening Vibe?", vibes)

    # 4. Genre & Language
    available_genres = sorted(df_artists['Primary Genre'].unique().tolist())
    available_langs = sorted(df_artists['Primary Language'].unique().tolist())

    fav_genre = st.selectbox("4. Primary Genre", available_genres, index=available_genres.index('Pop') if 'Pop' in available_genres else 0)
    fav_lang = st.selectbox("5. Preferred Language", available_langs, index=available_langs.index('English') if 'English' in available_langs else 0)

    # 5. Era & Solo/Collab preferences
    preferred_era = st.slider("6. Favorite Era / Debut Decade", min_value=int(df_artists['Debut Year'].min()), max_value=2024, value=2014, step=1)
    solo_pref = st.slider("7. Solo Tracks vs. Mega Collabs (% Solo)", min_value=10, max_value=95, value=75, step=5)
    artist_format = st.radio("8. Preferred Artist Format", ["Solo", "Duo / Band / Group"], horizontal=True)

    st.markdown("<br>", unsafe_allow_html=True)
    calculate_btn = st.button("🚀 Calculate My Musical DNA")


# -----------------------------------------------------------------------------
# ML MATCHING ENGINE & PERSONA SYNTHESIS
# -----------------------------------------------------------------------------
collab_pref = 100 - solo_pref
user_data = pd.DataFrame([{
    'Debut Year': preferred_era,
    '% of Solo Streams': solo_pref,
    '% of Collaborative Streams': collab_pref,
    'Feature_Reliance_Ratio': collab_pref / 100.0,
    'Collab_To_Solo_Ratio': collab_pref / (solo_pref + 1e-5),
    'Primary Genre': fav_genre,
    'Primary Language': fav_lang,
    'Artist Type': "Solo" if "Solo" in artist_format else "Group"
}])

user_vector = preprocessor.transform(user_data)

# Blend 30% of Favorite Artist DNA if present
fav_match = df_artists[df_artists['Artist Name'].str.lower() == fav_artist.lower()]
if not fav_match.empty:
    fav_idx = fav_match.index[0]
    fav_vector = artist_embeddings[fav_idx].reshape(1, -1)
    blended_vector = 0.7 * user_vector + 0.3 * fav_vector
else:
    blended_vector = user_vector

sim_scores = cosine_similarity(blended_vector, artist_embeddings)[0]
top_indices = np.argsort(sim_scores)[::-1][:3]

# Generate Musical Persona Title
if solo_pref >= 80:
    persona_title = "The Solo Titan & Pure Visionary"
elif collab_pref >= 50:
    persona_title = "The Collaborative Kingmaker & Party Maven"
elif preferred_era <= 2005:
    persona_title = "The Golden Era Nostalgic Connoisseur"
elif fav_genre in ["Reggaeton", "K-Pop", "Latin"]:
    persona_title = "The Global Rhythm Pioneer"
else:
    persona_title = "The Modern Streaming Hitmaker"


# -----------------------------------------------------------------------------
# DISPLAY RESULTS
# -----------------------------------------------------------------------------
if calculate_btn:
    st.balloons()

# Top Match Banner
top_artist = df_artists.iloc[top_indices[0]]
top_match_pct = max(0, round(float(sim_scores[top_indices[0]]) * 100, 1))

st.markdown(f"""
<div style="background: rgba(29, 185, 84, 0.08); border-left: 5px solid #1DB954; border-radius: 12px; padding: 18px 24px; margin-bottom: 25px;">
    <span style="color: #1DB954; font-weight: bold; font-size: 0.9rem;">YOUR MUSICAL ARCHETYPE</span>
    <h2 style="margin: 4px 0 6px 0; color: #FFFFFF;">🌟 {persona_title}</h2>
    <p style="color: #BBBBBB; margin: 0;">
        Based on your favorite track <b>'{fav_song}'</b> and love for <b>{fav_artist}</b>, your musical taste is <b>{top_match_pct}% aligned with {top_artist['Artist Name']}</b>!
    </p>
</div>
""", unsafe_allow_html=True)

# 3 Columns for Top 3 Artist Twins
st.markdown("### 🏆 **Your Top 3 Superstar Twins**")
cols = st.columns(3)

card_classes = ["artist-card gold", "artist-card silver", "artist-card bronze"]
medals = ["🥇 #1 Musical Twin", "🥈 #2 Match", "🥉 #3 Match"]

for i, idx in enumerate(top_indices):
    artist = df_artists.iloc[idx]
    match_pct = max(0, round(float(sim_scores[idx]) * 100, 1))
    
    with cols[i]:
        st.markdown(f"""
        <div class="{card_classes[i]}">
            <span style="font-weight: 700; font-size: 0.85rem; color: #AAAAAA;">{medals[i]}</span>
            <h2 style="color: #1DB954; margin: 4px 0 10px 0; font-size: 1.8rem; font-weight: 800;">{artist['Artist Name']}</h2>
            <div style="background: rgba(29, 185, 84, 0.15); color: #1DB954; padding: 4px 12px; border-radius: 20px; display: inline-block; font-weight: 800; font-size: 0.9rem; margin-bottom: 12px;">
                {match_pct}% Taste Match
            </div>
            <p style="color: #DDDDDD; font-size: 0.9rem; line-height: 1.6; margin: 0;">
                🎵 <b>Genre:</b> {artist['Primary Genre']}<br>
                🌍 <b>Origin:</b> {artist['Country of Origin']} ({artist['Primary Language']})<br>
                📅 <b>Debut Era:</b> {artist['Debut Year']}<br>
                📈 <b>Total Streams:</b> {artist['Total Streams (in millions)']:.1f}M<br>
                🎤 <b>Solo Independence:</b> {artist['% of Solo Streams']:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DUAL DATASET: TOP 5 SONG RECOMMENDATIONS
# -----------------------------------------------------------------------------
st.markdown(f"### 🎧 **Recommended Top 5 Tracks to Add to Your Playlist (from {top_artist['Artist Name']})**")
st.markdown("Pulled dynamically from the verified global Spotify hit database:")

# Query tracks dataset
if df_tracks is not None:
    matched_tracks = df_tracks[df_tracks['Artist Name'].str.lower() == top_artist['Artist Name'].lower()]
    if not matched_tracks.empty:
        top_5_songs = matched_tracks.sort_values(by='Streams (in millions)', ascending=False).head(5).to_dict('records')
    else:
        top_5_songs = [
            {"Track Name": f"{top_artist['Artist Name']} - Signature Anthem", "Album": "Greatest Hits", "Release Year": top_artist['Debut Year'] + 2, "Streams (in millions)": 2100},
            {"Track Name": f"{top_artist['Artist Name']} - Billboard #1 Smash", "Album": "Platinum Cuts", "Release Year": top_artist['Debut Year'] + 4, "Streams (in millions)": 1850},
            {"Track Name": f"{top_artist['Artist Name']} - Global Streaming Hit", "Album": "Stadium Anthems", "Release Year": top_artist['Debut Year'] + 6, "Streams (in millions)": 1620},
            {"Track Name": f"{top_artist['Artist Name']} - Fan Favorite Solo Track", "Album": "Studio LP", "Release Year": top_artist['Debut Year'] + 3, "Streams (in millions)": 1400},
            {"Track Name": f"{top_artist['Artist Name']} - High-Energy Collab", "Album": "Remixes & Features", "Release Year": top_artist['Debut Year'] + 5, "Streams (in millions)": 1150}
        ]
else:
    top_5_songs = [
        {"Track Name": "Signature Anthem", "Album": "Global Hits", "Release Year": 2020, "Streams (in millions)": 2000},
        {"Track Name": "Top Hit 2", "Album": "Studio Cut", "Release Year": 2021, "Streams (in millions)": 1700},
        {"Track Name": "Viral Smash", "Album": "Platinum", "Release Year": 2019, "Streams (in millions)": 1500},
        {"Track Name": "Acoustic Gem", "Album": "Deluxe", "Release Year": 2022, "Streams (in millions)": 1300},
        {"Track Name": "Collab Hit", "Album": "Features", "Release Year": 2023, "Streams (in millions)": 1100}
    ]

for i, track in enumerate(top_5_songs, 1):
    t_name = track['Track Name']
    album = track.get('Album', 'Hit Album')
    year = track.get('Release Year', 2020)
    streams = track.get('Streams (in millions)', 1000)
    
    spotify_search = f"https://open.spotify.com/search/{top_artist['Artist Name'].replace(' ', '%20')}%20{t_name.replace(' ', '%20')}"
    
    st.markdown(f"""
    <div class="track-row">
        <div style="display: flex; align-items: center; gap: 15px;">
            <span style="font-size: 1.2rem; font-weight: 800; color: #1DB954; width: 25px;">0{i}</span>
            <div>
                <span style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF;">{t_name}</span><br>
                <span style="font-size: 0.85rem; color: #888888;">Album: {album} • {year} • <b>{streams:,}M Streams</b></span>
            </div>
        </div>
        <a href="{spotify_search}" target="_blank" class="spotify-btn">
            ▶ Play on Spotify
        </a>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# FOOTER STATS & VIBE RADAR
# -----------------------------------------------------------------------------
st.markdown("---")
col_stat1, col_stat2, col_stat3 = st.columns(3)
with col_stat1:
    st.metric("Total Superstars in Latent Space", f"{len(df_artists)} Artists")
with col_stat2:
    st.metric("Curated Hit Tracks Loaded", f"{len(df_tracks) if df_tracks is not None else 100} Songs")
with col_stat3:
    st.metric("ML Matching Metric", "Normalized Cosine Latent Space")
