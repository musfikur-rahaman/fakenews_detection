import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from llmhelper import explain_fake_news
from dotenv import load_dotenv
import os
import torch
import torch.nn.functional as F
import re

# ---------- ENV ----------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Groq API key not found! Please set it in your .env or Streamlit secrets.")
    st.stop()

# ---------- CONFIG ----------
THRESHOLD = 0.55
MODEL_NAME = "afsanehm/fake-news-detection-llm"

# ---------- LOAD MODEL ----------
@st.cache_resource(show_spinner=False)
def load_classifier():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer

@st.cache_data(show_spinner=False)
def get_fake_explanation(text):
    return explain_fake_news(text)

# ---------- HELPER FUNCTIONS ----------
def map_label(label):
    return {
        "FAKE": "FAKE",
        "REAL": "REAL",
        "LABEL_1": "FAKE",
        "LABEL_0": "REAL"
    }.get(label, label)

def classify_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).squeeze().tolist()
    return {"REAL": probs[0], "FAKE": probs[1]}

def final_label(probs, threshold=THRESHOLD):
    return "FAKE" if probs["FAKE"] >= threshold else "REAL"

# ---------- HALLUCINATION DETECTORS ----------
def flag_impossible_medical_claim(text):
    impossible_patterns = [r'\bovernight\b', r'\binstantly\b', r'\bimmediately\b', r'\bin \d{1,2} days\b']
    miracle_words = [r'\bcures\b', r'\bheals\b', r'\breverses\b', r'\beliminates\b', r'\bmiracle\b']
    medical_terms = [r'\bdiabetes\b', r'\bcancer\b', r'\bheart disease\b', r'\bvirus\b', r'\binfection\b']
    pattern = '|'.join(impossible_patterns + miracle_words + medical_terms)
    return bool(re.search(pattern, text, flags=re.IGNORECASE))

def flag_absurd_or_fantasy_claim(text):
    absurd_keywords = [
        r'\balien\b', r'\bextraterrestrial\b', r'\btime travel\b', r'\bghost\b',
        r'\bmagic(al)?\b', r'\blevitate\b', r'\bsecretly married\b', r'\bshocking leak\b'
    ]
    pattern = '|'.join(absurd_keywords)
    return bool(re.search(pattern, text, flags=re.IGNORECASE))

def flag_impossible_numbers_or_timelines(text):
    patterns = [
        r'\b\d{7,}\b',
        r'\b\d{1,2} seconds\b', r'\b\d{1,2} minutes\b', r'\b\d{1,2} hours\b'
    ]
    pattern = '|'.join(patterns)
    return bool(re.search(pattern, text, flags=re.IGNORECASE))

def flag_financial_scams(text):
    patterns = [
        r'\$\d{3,} (daily|weekly|monthly)',
        r'no work required', r'get rich quick', r'earn \d{3,} per', r'unlimited money', r'free money'
    ]
    pattern = '|'.join(patterns)
    return bool(re.search(pattern, text, flags=re.IGNORECASE))

def flag_financial_exaggeration(text):
    patterns = [
        r'\b\d{2,4}%\b',
        r'\bafter .*tweet(s|ed)\b',
        r'\bmoon\b', r'\bhype\b', r'\bskyrocket\b', r'\bexplodes\b'
    ]
    pattern = '|'.join(patterns)
    return bool(re.search(pattern, text, flags=re.IGNORECASE))

def check_hallucination(text):
    return (
        flag_impossible_medical_claim(text) or
        flag_absurd_or_fantasy_claim(text) or
        flag_impossible_numbers_or_timelines(text) or
        flag_financial_scams(text) or
        flag_financial_exaggeration(text)
    )

# ---------- UI ----------
st.set_page_config(page_title="Fake News Detector", layout="centered")
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
if "probs" not in st.session_state: st.session_state.probs = None
if "hallucination_flag" not in st.session_state: st.session_state.hallucination_flag = None

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
    st.markdown("<h1>Fake News Detector</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Detect whether a news article is likely fake or real using AI.</div>", unsafe_allow_html=True)
    st.markdown("""
    **How to use:**
    1. Go to the **Classify** tab  
    2. Paste your news text  
    3. Click **Run Classification**  
    4. View prediction, confidence, and explanation  
    """)

# ---------- CLASSIFY PAGE ----------
elif st.session_state.page == "classify":
    st.markdown("<h1>Classify News</h1>", unsafe_allow_html=True)

    user_input = st.text_area(
        "Paste your news content below:",
        height=180,
        value=st.session_state.user_input,
        placeholder="e.g. Breaking: Scientists discover a new planet..."
    )

    col_reset, col_run = st.columns([1, 2])

    with col_reset:
        if st.button("Reset"):
            st.session_state.user_input = ""
            st.session_state.result = None
            st.session_state.probs = None
            st.session_state.hallucination_flag = None
            st.rerun()  # <--- updated for latest Streamlit

    with col_run:
        if st.button("Run Classification", type="primary", use_container_width=True):
            if user_input.strip():
                st.session_state.user_input = user_input
                with st.spinner("Analyzing..."):
                    model, tokenizer = load_classifier()
                    probs = classify_text(user_input, model, tokenizer)
                    label = final_label(probs, THRESHOLD)

                    hallucination_flag = check_hallucination(user_input)
                    if hallucination_flag:
                        label = "FAKE"
                        probs["FAKE"] = max(probs["FAKE"], 0.95)

                    st.session_state.result = label
                    st.session_state.probs = probs
                    st.session_state.hallucination_flag = hallucination_flag

    if st.session_state.result:
        color = "#ff4b4b" if st.session_state.result == "FAKE" else "#28a745"
        st.markdown(f"<div class='result-box' style='border-left: 6px solid {color}'>", unsafe_allow_html=True)
        st.subheader("Prediction Results")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Label", st.session_state.result)
        with col2: st.metric("FAKE Probability", f"{st.session_state.probs['FAKE']*100:.2f}%")
        with col3: st.metric("REAL Probability", f"{st.session_state.probs['REAL']*100:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("🔍 View Model Scores & Hallucination Flags"):
            st.json(st.session_state.probs)
            st.write(f"Hallucination / Impossible Claim Flag: {st.session_state.hallucination_flag}")

        if st.session_state.result == "FAKE":
            with st.spinner("Generating explanation..."):
                explanation = get_fake_explanation(user_input)
            st.markdown("#### 🧠 Why This May Be Fake")
            st.write(explanation)
        else:
            st.success("✅ This news was classified as REAL.")

# ---------- FOOTER ----------
st.markdown("""
<hr style="margin-top:40px;">
<div style="text-align:center; font-size:13px; color:#888;">
Built with Streamlit, HuggingFace Transformers, and Groq LLaMA.
</div>
""", unsafe_allow_html=True)
