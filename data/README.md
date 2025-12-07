# 📊 Dataset Overview

This project uses two datasets to build a **hybrid music recommendation system**:

- **Content-based dataset → Spotify Tracks Dataset**
- **User-based dataset → Last.fm User Listening History**

Both datasets serve different purposes:
- **Spotify** provides detailed audio features.
- **Last.fm** provides real user listening behaviour.

---

## 🎵 1. Last.fm User Dataset (User-Based Filtering)

This dataset contains user listening activity recorded throughout **January 2021**.  
It helps model user behaviour and preference patterns.

### 📌 Columns

| Column   | Description |
|----------|-------------|
| **Username** | Name/ID of the user |
| **Artist** | Artist the user listened to |
| **Track** | Name of the track |
| **Album** | Album containing the track |
| **Date** | Listening date (1 Jan – 31 Jan 2021) |
| **Time** | Exact timestamp when track was played |

### 📌 Purpose in Model

Used for **user-based collaborative filtering**:

- Learn repeated listening behaviour  
- Identify similar users  
- Predict what a user may like based on similar user patterns  

➡️ Implemented using the **PyTorch collaborative filtering model**.

---

## 🎧 2. Spotify Tracks Dataset (Content-Based Filtering)

This dataset contains rich **audio features** for each track via Spotify’s metadata.

### 📌 Columns

| Column | Description |
|--------|-------------|
| **track_id** | Spotify ID for the track |
| **artists** | Artist(s) performing the track |
| **album_name** | Album where the track appears |
| **track_name** | Track title |
| **popularity** | Popularity score (0–100) |
| **duration_ms** | Duration in milliseconds |
| **explicit** | Explicit content (true/false) |
| **danceability** | Dance suitability (0–1) |
| **energy** | Track intensity (0–1) |
| **key** | Musical key (0–11) |
| **loudness** | Overall loudness (dB) |
| **mode** | Major (1) or minor (0) |
| **speechiness** | Amount of spoken words (0–1) |
| **acousticness** | Likelihood track is acoustic (0–1) |
| **instrumentalness** | Likelihood track has no vocals (0–1) |
| **liveness** | Probability track is live (0–1) |
| **valence** | Positiveness of the track (0–1) |
| **tempo** | BPM (beats per minute) |
| **time_signature** | Time signature (3–7) |
| **track_genre** | Track genre |

### 📌 Purpose in Model

Used for **content-based filtering with Scikit-Learn**:

- Compute similarity between tracks  
- Recommend musically similar songs  
- Use features like valence, tempo, genre, danceability  

➡️ Implemented using **Scikit-Learn similarity models**.

---

## 🔗 How Both Datasets Work Together

This project uses a **hybrid recommendation system** combining:

- **70% Content-Based Score** (Spotify audio features)
- **30% User-Based Score** (Last.fm behaviour)

This ensures recommendations that consider:

- ✔ What the user likes  
- ✔ What songs sound similar  

---