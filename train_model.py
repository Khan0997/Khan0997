
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

texts = [
    "Это правда", 
    "Это фейк", 
    "Настоящая новость", 
    "Фейковая информация", 
    "Это вымысел", 
    "Подтверждённый факт"
]
labels = [0, 1, 0, 1, 1, 0]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

joblib.dump(vectorizer, "vectorizer.joblib")
joblib.dump(model, "model.joblib")
