import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Load data (Kaggle spam.csv: v1 = label, v2 = message)
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]
df["y"] = (df["label"] == "spam").astype(int)

# Split (stratified because spam is a minority class)
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["y"], test_size=0.2,
    random_state=42, stratify=df["y"])

# TF-IDF + logistic regression
model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                              stop_words="english", min_df=2)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
model.fit(X_train, y_train)

# Evaluate
print(classification_report(y_test, model.predict(X_test),
                            target_names=["ham", "spam"]))

joblib.dump(model, "sms_model.joblib")

# Interactive demo
print("\nType a message (or 'q' to quit):")
while True:
    msg = input("> ")
    if msg.strip().lower() == "q":
        break
    p = model.predict_proba([msg])[0][1]
    print(f"{'SCAM' if p > 0.5 else 'SAFE'} (scam probability: {p:.2f})")