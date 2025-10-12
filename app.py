import streamlit as st
from transformers import pipeline
from llmhelper import explain_fake_news
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Groq API key not found! Please set it in your .env or Streamlit secrets.")
    st.stop()

@st.cache_resource(show_spinner=False)
def load_classifier():
    return pipeline("text-classification", model="afsanehm/fake-news-detection-llm")

@st.cache_data(show_spinner=False)
def get_fake_explanation(text):
    return explain_fake_news(text)

def map_label(label):
    return {
        "FAKE": "FAKE",
        "REAL": "REAL",
        "LABEL_1": "FAKE",
        "LABEL_0": "REAL"
    }.get(label, label)

# Page config
st.set_page_config(page_title="Fake News Detector", layout="centered")

# Custom CSS
st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
            font-family: 'Segoe UI', sans-serif;
        }
        .nav-container {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-bottom: 2rem;
        }
        .nav-button {
            background-color: #1f4e79;
            color: white;
            border: none;
            padding: 0.6rem 1.8rem;
            border-radius: 6px;
            font-size: 1rem;
            cursor: pointer;
        }
        .nav-button:hover {
            background-color: #163b5c;
        }
        h1 {
            color: #1f4e79;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            text-align: center;
            font-size: 1.1rem;
            color: #555;
            margin-bottom: 2rem;
        }
        .result-box {
            background-color: #ffffff;
            border: 1px solid #ddd;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "home"
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Navigation buttons using session state
col1, col2 = st.columns(2)
with col1:
    if st.button("🏠 Home", use_container_width=True, key="home_btn"):
        st.session_state.page = "home"
        st.rerun()
with col2:
    if st.button("📋 Classify", use_container_width=True, key="classify_btn"):
        st.session_state.page = "classify"
        st.rerun()

st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)

# Home page
if st.session_state.page == "home":
    st.markdown("<h1>Fake News Detector</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Check if a news article is likely fake or real using AI.</div>", unsafe_allow_html=True)
    st.markdown("""
    This tool uses advanced AI models to analyze news content and determine whether it's likely to be fake or real.
    
    **How it works:**
    1. Paste your news content in the Classify tab
    2. Click "Run Classification"
    3. Get an instant prediction with confidence score
    4. For fake news, receive an AI-generated explanation
    """)

# Classify page
elif st.session_state.page == "classify":
    st.markdown("<h1>Classify News</h1>", unsafe_allow_html=True)

    user_input = st.text_area(
        "Paste your news content below:", 
        height=180, 
        value=st.session_state.user_input, 
        placeholder="e.g. Breaking: Scientists discover a new planet..."
    )

    if st.button("Run Classification", type="primary", use_container_width=True):
        if user_input.strip():
            st.session_state.user_input = user_input
            
            with st.spinner("Analyzing..."):
                clf = load_classifier()
                result = clf(user_input, truncation=True)[0]
                label = result['label']
                score = result['score']
                display_label = map_label(label)

            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.subheader("Prediction")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Label", display_label)
            with col2:
                st.metric("Confidence", f"{score:.2%}")
            st.markdown("</div>", unsafe_allow_html=True)

            if label in ["FAKE", "LABEL_1"]:
                with st.spinner("Generating explanation..."):
                    explanation = get_fake_explanation(user_input)
                st.markdown("#### Why This May Be Fake")
                st.write(explanation)
            else:
                st.info("✅ This news was classified as REAL.")
        else:
            st.warning("Please paste some news content to classify.")

# Footer
st.markdown("""
    <hr style="margin-top:40px;">
    <div style="text-align:center; font-size:13px; color:#888;">
        Built with Streamlit, HuggingFace Transformers, and Groq LLaMA.
    </div>
""", unsafe_allow_html=True)