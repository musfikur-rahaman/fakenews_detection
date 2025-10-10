import streamlit as st
from langchain_groq import ChatGroq

# Get the Groq API key from Streamlit secrets
API_KEY = st.secrets["GROQ_API_KEY"]

# Initialize the LLM
llm = ChatGroq(
    api_key=API_KEY,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

def explain_fake_news(text: str) -> str:
    """
    Use Groq LLaMA to explain why the input is considered fake news.
    """
    prompt = (
        "The following news article has been classified as FAKE.\n"
        "Explain in a few sentences why this might be fake news:\n\n"
        f"{text}\n\nExplanation:"
    )
    response = llm.invoke(prompt)
    return response.content.strip()

def explain_real_news(text: str) -> str:
    """
    Use Groq LLaMA to explain why the input is considered real/credible news.
    """
    prompt = (
        "The following news article has been classified as REAL.\n"
        "Explain in a few sentences why this might be credible and trustworthy news:\n\n"
        f"{text}\n\nExplanation:"
    )
    response = llm.invoke(prompt)
    return response.content.strip()
