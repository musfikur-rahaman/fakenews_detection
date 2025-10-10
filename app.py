import streamlit as st
from transformers import pipeline
from llmhelper import explain_fake_news  # your Groq explanation function
from dotenv import load_dotenv
import os

load_dotenv()  # load local .env if available

# Get API key from environment or Streamlit secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Groq API key not found! Please set it in your .env or Streamlit secrets.")
    st.stop()

@st.cache_resource(show_spinner=False)
def load_classifier():
    return pipeline("text-classification", model="afsanehm/fake-news-detection-llm")

@st.cache_data(show_spinner=False)
def get_explanation(text):
    return explain_fake_news(text)

def map_label(label):
    label_map = {
        "FAKE": "🚨 FAKE",
        "REAL": "✅ REAL",
        "LABEL_1": "🚨 FAKE",
        "LABEL_0": "✅ REAL"
    }
    return label_map.get(label, label)

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
def map_label(label):
        st.success(f"**Prediction:** {display_label}")
        st.write(f"**Confidence:** `{score:.2f}`")

        if label in ["FAKE", "LABEL_1"]:
            with st.spinner("Generating explanation with Groq LLaMA..."):
                explanation = get_explanation(user_input)
            with st.expander("Explanation why this might be fake"):
                st.write(explanation)
        else:
            st.info("This news was classified as REAL, no explanation generated.")

