import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]

df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Vectorization
vectorizer = CountVectorizer(
    stop_words="english",
    ngram_range=(1,2),
    lowercase=True
)
X = vectorizer.fit_transform(df['text'])
y = df["label"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model & vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model & Vectorizer saved")
