from tools import ml_sentiment_tool, llm_sentiment_tool

def agent_decide(text):
    ml_pred = ml_sentiment_tool(text)
    llm_pred = llm_sentiment_tool(text)

    # Convert to same case for comparison
    if ml_pred.lower() == llm_pred.lower():
        final = ml_pred
        reason = "Consensus Achieved: Both ML and Agentic logic agree."
    else:
        # If they disagree, we default to Neutral or flag for review
        final = "Neutral/Inconclusive"
        reason = f"Dispute: ML predicted {ml_pred}, but LLM logic suggested {llm_pred}."
    
    return final, reason

if __name__ == "__main__":
    sample_headlines = [
        "Company reports record profits this quarter",
        "Economic slowdown expected due to inflation",
        "Stock prices remain stable amid market uncertainty",
        "Nokia plans to expand its 5G infrastructure in Asia"
    ]
    
    print("AGENTIC SENTIMENT CLASSIFIER SYSTEM\n" + "="*40)
    for h in sample_headlines:
        sentiment, explanation = agent_decide(h)
        print(f"Headline: {h}")
        print(f"Final Decision: {sentiment}")
        print(f"Reason: {explanation}")
        print("-" * 50)
