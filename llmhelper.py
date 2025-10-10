
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()

# Get API key from env vars or from Streamlit secrets
API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

llm = ChatGroq(api_key=API_KEY, model_name="meta-llama/llama-4-scout-17b-16e-instruct")

def explain_fake_news(text):



    prompt = (
        f"The following news article has been classified as FAKE.\n"
        f"Explain in a few sentences why this might be fake news:\n\n{text}\n\nExplanation:"
    )
    response = llm.invoke(prompt)
    return response.content.strip()

if __name__ == "__main__":
    sample = "AI is taking over the world next year."
    print("Sample explanation:")
    print(explain_fake_news(sample))
