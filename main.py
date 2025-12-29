import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("games.csv")

df["combined"] = df["genres"] + " " + df["tags"]

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df["combined"])

similarity = cosine_similarity(tfidf_matrix)

def recommend(game_title, n=3):
    if game_title not in df["title"].values:
        return None
    
    
    idx = df[df["title"] == game_title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse = True)
    
    reccomendations = []
    
    for i in scores[1:n+1]:
        reccomendations.append({"title": df.iloc[i[0]]["title"], "score": round(i[1], 3)})

    return reccomendations

if __name__ == "__main__":
    user_game = input("Enter a game title: ")
    results = recommend(user_game)
    
    if results is None:
        print("Game not found. Please check the spelling.")
    else:
        print("\nRecommended games:")
        for r in results:
            print(f"{r['title']} - {r['score']}")  