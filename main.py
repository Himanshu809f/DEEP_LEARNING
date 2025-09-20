import nltk
from nltk.corpus import movie_reviews
import random
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Data download
nltk.download('movie_reviews')

# 2. Data Load
data = [
    (list(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]

# 3. Shuffle Data
random.shuffle(data)

# 4. Data Segregation
texts= [" ".join(words) for words, labels in data ]
labels = [label for words, label in data]

# 5. Data spliting
split = int(0.8 * len(texts))
train_texts, test_texts = texts[:split], texts[split:]
train_labels, test_labels = labels[:split], labels[split:]

# 6. Feature Extraction
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
x_train = vectorizer.fit_transform(train_texts)
x_test = vectorizer.transform(test_texts)

# 7. Training the model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, train_labels)

# 8. Evaluation
pred = model.predict(x_test)
print(f"Accuracy: {accuracy_score(test_labels, pred)}")
print(f"Classification report: \n {classification_report(test_labels, pred)}")

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")