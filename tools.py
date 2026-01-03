import joblib

model = joblib.load('ml_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

def ml_sentiment_tool(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return pred

def llm_sentiment_tool(text):

    text_lower = text.lower()
    if 'profit' in text_lower or 'gain' in text_lower:
        return 'Positive'
    elif 'loss' in text_lower or 'decline' in text_lower:
        return 'Negative'
    else:
        return 'Neutral'
