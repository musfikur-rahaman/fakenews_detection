import streamlit as st
from transformers import pipeline
from llmhelper import explain_fake_news  # import your Groq explanation function
from dotenv import load_dotenv

load_dotenv()  # make sure env vars like GROQ_API_KEY are loaded

@st.cache_resource
def load_classifier():
    return pipeline("text-classification", model="afsanehm/fake-news-detection-llm")

def map_label(label):
    label_map = {"FAKE": "🚨 FAKE", "REAL": "✅ REAL", "LABEL_1": "🚨 FAKE", "LABEL_0": "✅ REAL"}
    return label_map.get(label, label)

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

        if label == "FAKE" or label == "LABEL_1":  # handle different label schemes
            with st.spinner("Generating explanation with Groq LLaMA..."):
                explanation = explain_fake_news(user_input)
            st.markdown("**Explanation why this might be fake:**")
            st.write(explanation)
        else:
            st.info("This news was classified as REAL, no explanation generated.")

