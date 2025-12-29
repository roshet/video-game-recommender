import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

df = pd.read_csv("games.csv")

df["combined"] = df["genres"] + " " + df["genres"] + " " + df["tags"]

tfidf = TfidfVectorizer(
    stop_words = "english",
    ngram_range = (1,2),
    min_df = 2
)

tfidf_matrix = tfidf.fit_transform(df["combined"])
similarity = cosine_similarity(tfidf_matrix)

feature_names = tfidf.get_feature_names_out()

GENERIC_TERMS = {"game", "games", "play", "player", "players", "single", "online", "multiplayer"}

def explain_similarity(idx1, idx2, top_n = 6):
    vec1 = tfidf_matrix[idx1].toarray()[0]
    vec2 = tfidf_matrix[idx2].toarray()[0]
    
    contributions = vec1 * vec2
    nonzero = contributions.nonzero()[0]
    
    feature_scores = [(feature_names[i], contributions[i]) for i in nonzero if feature_names[i] not in GENERIC_TERMS]
    
    feature_scores.sort(key = lambda x: x[1], reverse = True)
    
    feature_scores.sort(key = lambda x: len(x[0].split()), reverse = True)
    
    return feature_scores[:top_n]

GOOD_BIGRAMS = {"open world", "action rpg", "story rich", "single player", "role playing"}

def clean_feature_scores(feature_scores):
    cleaned = []
    
    for feature, score in feature_scores:
        words = feature.split()
        
        if len(words) == 2 and words[0] == words[1]:
            continue
        
        if len(words) == 2 and feature not in GOOD_BIGRAMS:
            continue
        
        cleaned.append((feature, score))
    
    return cleaned

def build_explanation(feature_scores, max_features = 3):
    features = [f for f, _ in feature_scores[:max_features]]
    
    if not features:
        return None
    
    if len(features) == 1:
        return f"Both games strongly emphasize {features[0]}."
    elif len(features) == 2:
        return f"Both games emphasize {features[0]} and {features[1]}."
    else:
        return (
            f"Both games emphasize {features[0]}, "
            f"{features[1]}, and {features[2]}"
            )

def recommend(game_title, n=3):
    if game_title not in df["title"].values:
        return None
    
    
    idx = df[df["title"] == game_title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse = True)
    
    recommendations = []
    
    for i in scores[1:n+1]:
        features = explain_similarity(idx, i[0])
        features = clean_feature_scores(features)
        recommendations.append({"title": df.iloc[i[0]]["title"], "score": round(i[1], 3), "features": features, "reason": build_explanation(features)})

    return recommendations


st.title("🎮 Video Game Recommendation System")
st.write("Select a game to receive similar game recommendations based on genres and tags.")

game_list = df["title"].tolist()
selected_game = st.selectbox("Select a game:", game_list)

if st.button("Get Recommendations"):
    st.markdown("## 🔍 Recommened Games")
    results = recommend(selected_game)
    
    if results is None:
        st.error("Game not found.")
    else:
        for r in results:
            with st.container():
                st.markdown(f"### {r['title']}")
                st.write(f"**Similarity score:** {r['score']:.2f}")
            
                if r["reason"]:
                    st.write("**Why this was recommended:**")
                    st.write(r["reason"])
                
                st.caption("Key similarities:**")
                for f, _ in r["features"]:
                    st.write(f"• {f}")
            
            st.markdown("---")