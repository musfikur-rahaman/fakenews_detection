import os
import streamlit as st
from dotenv import load_dotenv
from transformers import pipeline
from langchain_groq import ChatGroq

load_dotenv()

# Get API key from env vars or from Streamlit secrets
API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

llm = ChatGroq(api_key=API_KEY, model_name="meta-llama/llama-4-scout-17b-16e-instruct")

@st.cache_resource
def load_classifier():
    return pipeline("text-classification", model="afsanehm/fake-news-detection-llm")

def map_label(label):
    label_map = {
        "FAKE": "🚨 FAKE",
        "REAL": "✅ REAL",
        "LABEL_1": "🚨 FAKE",
        "LABEL_0": "✅ REAL"
    }
    return label_map.get(label, label)

def explain_fake_news(text):
    prompt = (
        f"The following news article has been classified as FAKE.\n"
        f"Explain in a few sentences why this might be fake news:\n\n{text}\n\nExplanation:"
    )
    response = llm.invoke(prompt)
    return response.content.strip()

def explain_real_news(text):
    prompt = (
        f"The following news article has been classified as REAL.\n"
        f"Explain in a few sentences why this news appears credible and reliable:\n\n{text}\n\nExplanation:"
    )
    response = llm.invoke(prompt)
    return response.content.strip()

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector with Explanation")
st.markdown("Enter a news article, headline, or paragraph to check whether it's likely **fake or real**.")

user_input = st.text_area("Enter news content here:", height=200)

if st.button("🔍 Classify"):
    if not user_input.strip():
        st.warning("Please enter some text to classify.")
    else:
        with st.spinner("Classifying..."):
            clf = load_classifier()
            result = clf(user_input, truncation=True)[0]
            label = result['label']
            score = result['score']
            display_label = map_label(label)

        st.success(f"**Prediction:** {display_label}")
        st.write(f"**Confidence:** `{score:.2f}`")

        if label == "FAKE" or label == "LABEL_1":
            with st.spinner("Generating explanation with Groq LLaMA..."):
                explanation = explain_fake_news(user_input)
            st.markdown("**Explanation why this might be fake:**")
            st.write(explanation)
        else:
            with st.spinner("Generating explanation for real news..."):
                explanation = explain_real_news(user_input)
            st.markdown("**Explanation why this news appears credible:**")
            st.write(explanation)
