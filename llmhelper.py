import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Get API key from env vars or Streamlit secrets
API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

llm = ChatGroq(api_key=API_KEY, model_name="meta-llama/llama-4-scout-17b-16e-instruct")

def explain_fake_news(text):
    """
    Generate explanation for why a news article may be FAKE.
    """
    prompt = (
        f"The following news article has been classified as FAKE.\n"
        f"Explain in a few sentences why this might be fake news:\n\n{text}\n\nExplanation:"
    )
    response = llm.invoke(prompt)
    return response.content.strip()

def fact_check(text):
    """
    Classify news as FAKE or REAL using LLM ensemble.
    Always returns 'FAKE' or 'REAL'.
    """
    prompt = (
        f"Classify the following news as FAKE or REAL. "
        f"Treat obviously absurd, humorous, impossible, or sensational claims as FAKE. "
        f"Answer ONLY with FAKE or REAL.\n\nNews: {text}\n\nAnswer:"
    )
    response = llm.invoke(prompt)
    label = response.content.strip().upper()
    if "FAKE" in label:
        return "FAKE"
    else:
        return "REAL"


if __name__ == "__main__":
    sample = "Emotional support clown hired during company layoffs."
    print("Sample explanation:")
    print(explain_fake_news(sample))
    print("Fact check:", fact_check(sample))
