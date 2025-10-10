import streamlit as st
from transformers import pipeline
from llmhelper import explain_fake_news, explain_real_news

# Load GROQ API key from Streamlit secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

if not GROQ_API_KEY:
    st.error("Groq API key not found in Streamlit secrets.")
    st.stop()

@st.cache_resource(show_spinner=False)
def load_classifier():
    return pipeline("text-classification", model="afsanehm/fake-news-detection-llm")

@st.cache_data(show_spinner=False)
def get_fake_explanation(text):
    return explain_fake_news(text)

@st.cache_data(show_spinner=False)
def get_real_explanation(text):
    return explain_real_news(text)

def map_label(label):
    label_map = {
        "FAKE": "🚨 FAKE",
        "REAL": "✅ REAL",
        "LABEL_1": "🚨 FAKE",
        "LABEL_0": "✅ REAL"
    }
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

        if label in ["FAKE", "LABEL_1"]:
            with st.spinner("Generating explanation with Groq LLaMA..."):
                explanation = get_fake_explanation(user_input)
            with st.expander("Explanation why this might be fake"):
                st.write(explanation)
        else:
            with st.spinner("Generating explanation for real news with Groq LLaMA..."):
                explanation = get_real_explanation(user_input)
            with st.expander("Explanation why this news appears credible"):
                st.write(explanation)
