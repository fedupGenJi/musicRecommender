from content_recommender import recommend_content
from user_recommender import recommend_for_user

def recommend_hybrid(track_name, username, top_n=10):
    content_recs = recommend_content(track_name, top_n=50)
    user_recs = recommend_for_user(username, top_n=50)

    content_tracks = {
        r["track_name"].lower(): r for r in content_recs
    }

    user_tracks = {
        r["track"].lower(): r for r in user_recs
    }

    common = set(content_tracks.keys()) & set(user_tracks.keys())

    results = []

    for track in common:
        results.append({
            "track_name": content_tracks[track]["track_name"],
            "artist": content_tracks[track].get("artist", ""),
            "content_score": content_tracks[track].get("similarity", 0),
            "user_score": user_tracks[track].get("score", 0),
            "hybrid_score": (
                0.6 * content_tracks[track].get("similarity", 0)
                + 0.4 * user_tracks[track].get("score", 0)
            )
        })

    if not results:
        results = [
            {
                "track_name": r["track_name"],
                "artist": r.get("artist", ""),
                "hybrid_score": r.get("similarity", 0)
            }
            for r in content_recs[:top_n]
        ]

    return sorted(results, key=lambda x: x["hybrid_score"], reverse=True)[:top_n]

if __name__ == "__main__": 
    recommendations = recommend_hybrid("Yellow", "isaac", top_n=10) 
    for r in recommendations: 
        print(r)