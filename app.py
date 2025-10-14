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

MODEL_NAME = "afsanehm/fake-news-detection-llm"
THRESHOLD = 0.55

# ---------- LOAD MODEL ----------
@st.cache_resource(show_spinner=False)
def load_classifier():
    return pipeline("text-classification", model=MODEL_NAME)

@st.cache_data(show_spinner=False)
def get_fake_explanation(text):
    return explain_fake_news(text)

# ---------- ENHANCED HALLUCINATION CHECK ----------
def flag_hallucination_keywords(text, extra_keywords=None):
    patterns = [
        # medical miracles / instant cures
        r'\bovernight\b|\binstantly\b|\bcures?\b|\bmiracle\b|\bdiabetes\b|\bcancer\b|\bAIDS\b',
        # aliens, paranormal, supernatural
        r'\balien\b|\bextraterrestrial\b|\bufo\b|\bpyramid\b|\bghost\b|\btelepathic\b',
        # money & scams
        r'\$\d{2,}|\bget rich\b|\bno work\b|\bpassive income\b|\bwin big\b|\bjackpot\b|\bmake money fast\b',
        # celebrity / drama / scandal
        r'\bcelebrity\b|\bsecretly married\b|\bshocking\b|\bscandal\b|\bleaked\b',
        # government & cover-ups
        r'\bcover(ing)? up\b|\bsecret program\b|\bclassified\b|\bdeep state\b|\bCIA\b|\bFBI\b|\bNASA\b|\bgovernment hiding\b',
        # conspiracy & pseudoscience
        r'\b5G\b|\bvirus\b|\bvaccine\b|\bmicrochip\b|\bbioweapon\b|\bchemtrail\b|\bclimate hoax\b|\bflat earth\b',
        r'\bscientists warn\b|\bexperts say\b|\bhidden truth\b|\bthey don’t want you to know\b|\bspread(s)? deadly\b|\bnew study proves\b',
    ]
    if extra_keywords:
        patterns.append(extra_keywords)
    return bool(re.search('|'.join(patterns), text, flags=re.IGNORECASE))

# ---------- HYBRID CLASSIFICATION ----------
def hybrid_classify(text, classifier, extra_keywords=None):
    result = classifier(text, truncation=True)[0]
    label = result["label"]
    score = result["score"]
    halluc_flag = flag_hallucination_keywords(text, extra_keywords)
    if halluc_flag:
        label = "FAKE"
        score = max(score, 0.97)
    return label, score, halluc_flag

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Adaptive Fake News Detector", layout="centered")
st.markdown("""
<style>
body { background-color: #f5f7fa; font-family: 'Segoe UI', sans-serif; }
h1 { color: #1f4e79; text-align: center; margin-bottom: 0.5rem; }
.subtitle { text-align: center; font-size: 1.1rem; color: #555; margin-bottom: 2rem; }
.result-box { background-color: #ffffff; border: 1px solid #ddd; padding: 1rem; border-radius: 8px; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ---------- SESSION ----------
if "user_input" not in st.session_state: st.session_state.user_input = ""
if "history" not in st.session_state: st.session_state.history = []
if "feedback_keywords" not in st.session_state: st.session_state.feedback_keywords = []  # adaptive learning

# ---------- HEADER ----------
st.title("Adaptive Fake News Detector")
st.markdown("<div class='subtitle'>AI + hallucination + session learning</div>", unsafe_allow_html=True)

user_input = st.text_area("Enter news text:", height=180, placeholder="e.g. 5G Towers Found to Spread Deadly Virus, Scientists Warn", value=st.session_state.user_input)

col1, col2 = st.columns([1,2])
with col1:
    if st.button("Reset"):
        st.session_state.user_input = ""
        st.rerun()
with col2:
    if st.button("Run Classification", type="primary"):
        if user_input.strip():
            st.session_state.user_input = user_input
            with st.spinner("Analyzing..."):
                classifier = load_classifier()
                # Join adaptive session keywords into regex OR pattern
                extra_keywords = '|'.join(st.session_state.feedback_keywords) if st.session_state.feedback_keywords else None
                label, score, halluc_flag = hybrid_classify(user_input, classifier, extra_keywords)

            st.session_state.history.append({
                "News": user_input,
                "Label": label,
                "Confidence": f"{score*100:.2f}%",
                "Hallucination": halluc_flag
            })

            color = "#ff4b4b" if label == "FAKE" else "#28a745"
            st.markdown(f"<div class='result-box' style='border-left:6px solid {color}'>", unsafe_allow_html=True)
            st.subheader(f"Prediction: {label}")
            st.metric("Confidence", f"{score*100:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)

            if label == "FAKE":
                with st.spinner("Generating explanation..."):
                    explanation = get_fake_explanation(user_input)
                st.markdown("#### 🧠 Why This May Be Fake")
                st.write(explanation)
            else:
                st.success("✅ This news was classified as REAL.")

            # ---------- FEEDBACK ----------
            st.markdown("#### ✅ Is this prediction correct?")
            feedback = st.radio("Correct label?", ["Yes", "No"], key=f"feedback_{len(st.session_state.history)}")

            if feedback == "No":
                correct_label = st.selectbox("Correct label should be:", ["FAKE", "REAL"], key=f"correct_{len(st.session_state.history)}")
                if st.button("Update Prediction", key=f"update_{len(st.session_state.history)}"):
                    st.session_state.history[-1]["Label"] = correct_label
                    # If user says FAKE but model missed, add keywords for adaptive learning
                    if correct_label == "FAKE":
                        new_keywords = re.findall(r'\b\w+\b', user_input)
                        st.session_state.feedback_keywords.extend(new_keywords)
                    st.success("✅ Prediction updated and session learning applied!")

# ---------- SESSION HISTORY ----------
if st.session_state.history:
    st.markdown("### 📝 Session History")
    st.table(st.session_state.history)

st.markdown("""
<hr style="margin-top:40px;">
<div style="text-align:center; font-size:13px; color:#888;">
Built with Streamlit, HuggingFace Transformers, and Groq LLaMA.
</div>
""", unsafe_allow_html=True)
