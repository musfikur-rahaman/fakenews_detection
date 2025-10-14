import streamlit as st
from transformers import pipeline
from llmhelper import explain_fake_news
from dotenv import load_dotenv
import os
import re

# ---------- ENV ----------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Groq API key not found! Please set it in your .env or Streamlit secrets.")
    st.stop()

# ---------- CONFIG ----------
MODEL_NAME = "afsanehm/fake-news-detection-llm"
THRESHOLD = 0.55

# ---------- LOAD MODEL ----------
@st.cache_resource(show_spinner=False)
def load_classifier():
    return pipeline("text-classification", model=MODEL_NAME)

@st.cache_data(show_spinner=False)
def get_fake_explanation(text):
    return explain_fake_news(text)

# ---------- HALLUCINATION DETECTORS ----------
def flag_hallucination_keywords(text):
    patterns = [
        r'\bovernight\b|\binstantly\b|\bcures\b|\bmiracle\b|\bdisease\b',
        r'\balien\b|\bextraterrestrial\b|\btime travel\b|\bghost\b|\bpyramid\b',
        r'\$\d{2,} (daily|weekly|monthly)|get rich quick|no work required|stock \d{2,4}%',
        r'\bcelebrity\b|\bsecretly married\b|\bshocking\b|\bscandal\b',
        r'\bcover(ing)? up\b|\bsecret\b|\bgovernment\b|\bNASA\b|\bunexplained\b'
    ]
    return bool(re.search('|'.join(patterns), text, flags=re.IGNORECASE))

# ---------- HYBRID CLASSIFICATION ----------
def hybrid_classify(text, classifier):
    result = classifier(text, truncation=True)[0]
    label = result["label"]
    score = result["score"]
    halluc_flag = flag_hallucination_keywords(text)
    if halluc_flag:
        label = "FAKE"
        score = max(score, 0.95)
    return label, score, halluc_flag

# ---------- UI ----------
st.set_page_config(page_title="Hybrid Fake News Detector", layout="centered")
st.markdown("""
<style>
body { background-color: #f5f7fa; font-family: 'Segoe UI', sans-serif; }
h1 { color: #1f4e79; text-align: center; margin-bottom: 0.5rem; }
.subtitle { text-align: center; font-size: 1.1rem; color: #555; margin-bottom: 2rem; }
.result-box { background-color: #ffffff; border: 1px solid #ddd; padding: 1rem; border-radius: 8px; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if "page" not in st.session_state: st.session_state.page = "home"
if "user_input" not in st.session_state: st.session_state.user_input = ""
if "result" not in st.session_state: st.session_state.result = None
if "score" not in st.session_state: st.session_state.score = None
if "halluc_flag" not in st.session_state: st.session_state.halluc_flag = None
if "history" not in st.session_state: st.session_state.history = []

# ---------- NAVIGATION ----------
col1, col2 = st.columns(2)
with col1:
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
with col2:
    if st.button("📋 Classify", use_container_width=True):
        st.session_state.page = "classify"
        st.rerun()
st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)

# ---------- HOME PAGE ----------
if st.session_state.page == "home":
    st.markdown("<h1>Hybrid Fake News Detector</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Detect whether news is likely fake using AI and hallucination checks.</div>", unsafe_allow_html=True)
    st.markdown("""
**How to use:**
1. Go to the **Classify** tab  
2. Paste your news text  
3. Click **Run Classification**  
4. View prediction, confidence, explanation, and session history
""")

# ---------- CLASSIFY PAGE ----------
elif st.session_state.page == "classify":
    st.markdown("<h1>Classify News</h1>", unsafe_allow_html=True)

    user_input = st.text_area(
        "Paste your news content below:",
        height=180,
        value=st.session_state.user_input,
        placeholder="e.g. Ancient Pyramid Found on Mars — NASA Trying to Cover It Up"
    )

    col_reset, col_run = st.columns([1,2])
    with col_reset:
        if st.button("Reset"):
            st.session_state.user_input = ""
            st.session_state.result = None
            st.session_state.score = None
            st.session_state.halluc_flag = None
            st.rerun()

    with col_run:
        if st.button("Run Classification", type="primary", use_container_width=True):
            if user_input.strip():
                st.session_state.user_input = user_input
                with st.spinner("Analyzing..."):
                    classifier = load_classifier()
                    label, score, halluc_flag = hybrid_classify(user_input, classifier)

                    st.session_state.result = label
                    st.session_state.score = score
                    st.session_state.halluc_flag = halluc_flag

                    # Add to session history
                    st.session_state.history.append({
                        "News": user_input,
                        "Label": label,
                        "Confidence": f"{score*100:.2f}%",
                        "Hallucination Flag": halluc_flag
                    })

    # ---------- RESULTS ----------
    if st.session_state.result:
        color = "#ff4b4b" if st.session_state.result=="FAKE" else "#28a745"
        st.markdown(f"<div class='result-box' style='border-left: 6px solid {color}'>", unsafe_allow_html=True)
        st.subheader("Prediction Results")
        st.metric("Label", st.session_state.result)
        st.metric("Confidence", f"{st.session_state.score*100:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.result=="FAKE":
            with st.spinner("Generating explanation..."):
                explanation = get_fake_explanation(user_input)
            st.markdown("#### 🧠 Why This May Be Fake")
            st.write(explanation)
        else:
            st.success("✅ This news was classified as REAL.")

    # ---------- SESSION HISTORY ----------
    if st.session_state.history:
        st.markdown("### 📝 Session History")
        st.table(st.session_state.history)

# ---------- FOOTER ----------
st.markdown("""
<hr style="margin-top:40px;">
<div style="text-align:center; font-size:13px; color:#888;">
Built with Streamlit, HuggingFace Transformers, and Groq LLaMA.
</div>
""", unsafe_allow_html=True)
