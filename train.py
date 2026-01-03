import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import joblib

# Load dataset - Fixed column mapping
df = pd.read_csv('data/FinancialPhraseBank.csv', header=None, encoding='latin-1', names=['sentiment', 'text'])
df['sentiment'] = df['sentiment'].str.strip().str.lower()

# Split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# Pipeline create chesthunnam (Vectorizer + Model okate file lo save avthundi)
# Change ngram_range from (1, 2) to (1, 3)
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 3))), 
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])
# Train
model_pipeline.fit(X_train, y_train)

# Evaluate
y_pred = model_pipeline.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save simplified pipeline
joblib.dump(model_pipeline, 'financial_model.joblib')
print("Model Saved Successfully!")
