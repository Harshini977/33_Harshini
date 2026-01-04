import streamlit as st
import joblib
import requests
from tools import llm_sentiment_tool

# 1. Page Configuration
st.set_page_config(page_title="Agentic Finance AI", page_icon="🏦", layout="wide")

# 2. Load the trained ML Model
@st.cache_resource
def load_model():
    # Ensure this matches the filename from your train.py
    return joblib.load('financial_model.joblib')

try:
    pipeline = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please run 'py train.py' first.")

# 3. News Fetching Function
def fetch_live_news():
    # Your NewsAPI Key
    API_KEY = "4d56e0c1b2154c119d3b2e034e13c8ed"
    url = f"https://newsapi.org/v2/top-headlines?category=business&language=en&apiKey={API_KEY}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get('articles', [])[:8] # Get top 8 news items
        else:
            st.error(f"API Error: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return []

# 4. UI Header
st.title("🏦 Agentic Financial Sentiment Dashboard")
st.markdown("""
    This system uses a **Dual-Verification Agent**:
    1. **ML Layer:** Logistic Regression (trained on FinancialPhraseBank).
    2. **Agentic Layer:** Rule-based reasoning for cross-validation.
""")

# 5. Sidebar / Manual Analysis
st.sidebar.header("Manual Analysis")
user_input = st.sidebar.text_area("Type a headline to test:")
if st.sidebar.button("Analyze Headline"):
    if user_input:
       prediction = pipeline.predict([user_input])[0]
        st.sidebar.info(f"Predicted Sentiment: **{prediction.upper()}**")
    else:
        st.sidebar.warning("Please enter text first.")

# 6. Main Dashboard: Live Agent
st.header("📡 Live Market Monitor")
if st.button("Run Agent: Fetch & Cross-Verify Live News"):
    articles = fetch_live_news()
    
    if articles:
        # Create a grid for the news items
        cols = st.columns(2)
        for idx, art in enumerate(articles):
            title = art['title']
            if not title: continue
            
            # Get Predictions
            ml_pred = pipeline.predict([title])[0]
            agent_logic = llm_sentiment_tool(title)
            
            # Display in alternating columns
            with cols[idx % 2]:
                with st.container(border=True):
                    st.subheader(f"News #{idx+1}")
                    st.write(f"**{title}**")
                    
                    m1, m2 = st.columns(2)
                    m1.metric("ML Model", ml_pred.capitalize())
                    m2.metric("Agent Logic", agent_logic)
                    
                    if ml_pred.lower() == agent_logic.lower():
                        st.success("✅ Consensus Reached")
                    else:
                        st.warning("⚠️ Systems Disagree")
    else:
        st.info("No news articles found. Try again in a moment.")

st.divider()
st.caption("Agentic Sentiment Classifier | A Neoncoders Project")
