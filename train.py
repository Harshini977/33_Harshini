import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv('data/FinancialPhraseBank.csv', header=None)
df.columns = ['text', 'sentiment']

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)


vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

joblib.dump(model, 'ml_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

print("Model training complete and saved.")
