import os
import streamlit as st
from dotenv import load_dotenv
import types
import langchain

# ⚡ Patch missing langchain attributes required by ChatGroq
if not hasattr(langchain, "debug"):
    langchain.debug = types.SimpleNamespace(verbose=False)
if not hasattr(langchain, "llm_cache"):
    langchain.llm_cache = types.SimpleNamespace()

from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

# Initialize ChatGroq LLM
llm = ChatGroq(
    api_key=API_KEY,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.1,
    max_tokens=500
)

# Debug helper
def llm_invoke_with_debug(prompt: str) -> str:
    st.text_area("LLM Prompt", prompt, height=200)
    try:
        response = llm.invoke(prompt)
        st.text_area("LLM Response", response.content, height=200)
        return response.content.strip()
    except Exception as e:
        st.error(f"LLM invocation error: {e}")
        return f"Error: {e}"

# Fake news explanation
def explain_fake_news(text: str) -> str:
    prompt = (
        "Analyze the following news content and explain why it might be fake news. "
        "Focus on:\n"
        "1. Logical inconsistencies\n"
        "2. Sensational language\n"
        "3. Lack of credible sources\n"
        "4. Common fake news patterns\n"
        "5. Verification recommendations\n\n"
        f"Content: {text[:800]}\n\n"
        "Provide a balanced explanation in 3-4 sentences:"
    )
    return llm_invoke_with_debug(prompt)

# Fact-check
def fact_check(text: str) -> str:
    prompt = (
        "Determine if this news content is FAKE or REAL.\n"
        "Consider accuracy, sources, evidence, and sensationalism.\n\n"
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
        return "REAL"

# LLM explanation
def get_llm_explanation(text: str, classification: str) -> str:
    if classification == "FAKE":
        return explain_fake_news(text)
    else:
        prompt = (
            "Explain why this news content appears to be REAL and credible.\n"
            "Consider consistency, evidence, tone, and lack of fake news patterns.\n\n"
            f"Content: {text[:800]}\n\n"
            "Provide a brief explanation in 2-3 sentences:"
        )
        return llm_invoke_with_debug(prompt)

# Streamlit UI
st.title("Fake News Analysis with ChatGroq")

news_input = st.text_area("Enter news content:", height=150)

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
