import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Get API key from environment or Streamlit secrets
API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

# Initialize ChatGroq LLM
llm = ChatGroq(
    api_key=API_KEY,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.1,
    max_tokens=500
)

# Helper function to debug prompts and responses
def llm_invoke_with_debug(prompt: str) -> str:
    """
    Sends a prompt to the LLM and prints debugging info.
    """
    st.text_area("LLM Prompt", prompt, height=200)
    try:
        response = llm.invoke(prompt)
        st.text_area("LLM Response", response.content, height=200)
        return response.content.strip()
    except Exception as e:
        st.error(f"LLM invocation error: {e}")
        return f"Error: {e}"

# Generate explanation why news might be fake
def explain_fake_news(text: str) -> str:
    prompt = (
        "Analyze the following news content and explain why it might be fake news. "
        "Focus on:\n"
        "1. Logical inconsistencies or impossibilities\n"
        "2. Sensational or emotional language\n"
        "3. Lack of credible sources or evidence\n"
        "4. Common fake news patterns\n"
        "5. Recommendations for verification\n\n"
        f"Content: {text[:800]}\n\n"
        "Provide a balanced, factual explanation in 3-4 sentences:"
    )
    return llm_invoke_with_debug(prompt)

# Determine if news is FAKE or REAL
def fact_check(text: str) -> str:
    prompt = (
        "Carefully analyze this news content and determine if it's FAKE or REAL.\n"
        "Consider:\n"
        "- Factual accuracy and plausibility\n"
        "- Source credibility (if mentioned)\n"
        "- Evidence and specifics provided\n"
        "- Common misinformation patterns\n"
        "- Sensationalism vs factual reporting\n\n"
        f"Content: {text[:1000]}\n\n"
        "Answer ONLY with FAKE or REAL."
    )
    response = llm_invoke_with_debug(prompt)
    label = response.upper()
    if "FAKE" in label:
        return "FAKE"
    elif "REAL" in label:
        return "REAL"
    else:
        return "REAL"  # default fallback

# Provide explanation for either FAKE or REAL news
def get_llm_explanation(text: str, classification: str) -> str:
    if classification == "FAKE":
        return explain_fake_news(text)
    else:
        prompt = (
            "Explain why this news content appears to be REAL and credible.\n"
            "Consider:\n"
            "- Factual consistency\n"
            "- Plausible claims with evidence\n"
            "- Professional tone and language\n"
            "- Lack of common fake news patterns\n\n"
            f"Content: {text[:800]}\n\n"
            "Provide a brief explanation in 2-3 sentences:"
        )
        return llm_invoke_with_debug(prompt)

# Streamlit UI
st.title("Fake News Analysis with ChatGroq")

news_input = st.text_area("Enter news content for analysis:", height=150)

if st.button("Analyze News"):
    if not news_input.strip():
        st.warning("Please enter some news content.")
    else:
        st.subheader("Fact Check Result")
        classification = fact_check(news_input)
        st.success(f"Classification: {classification}")

        st.subheader("LLM Explanation")
        explanation = get_llm_explanation(news_input, classification)
        st.write(explanation)
