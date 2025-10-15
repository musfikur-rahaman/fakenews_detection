import streamlit as st
from transformers import pipeline
from llmhelper import explain_fake_news, fact_check
from source_validator import (
    check_source_reputation, 
    get_source_score, 
    analyze_url_characteristics,
    extract_domain
)
import re   

# ---------- MODEL LOADING ----------
@st.cache_resource(show_spinner=False)
def load_classifier():
    return pipeline("text-classification", model="afsanehm/fake-news-detection-llm")

# ---------- LABEL MAPPING ----------
def map_label(label):
    return {
        "LABEL_0": "REAL",
        "LABEL_1": "FAKE",
        "REAL": "REAL",
        "FAKE": "FAKE"
    }.get(label, label)

# ---------- HYBRID CLASSIFICATION WITH SOURCE VALIDATION ----------
def hybrid_classify(text, classifier, source_url=None):
    result = classifier(text, truncation=True)[0]
    label = map_label(result['label'])
    score = result['score']

    # Expanded hallucination keywords
    hallucination_keywords = [
        "microchip", "tracking", "cover-up", "hoax", "alien",
        "mind control", "flat earth", "5g", "secret experiment",
        "cure overnight", "steal the election", "cloned celebrity",
        "squirrel", "arrested animals", "talking animals", "flying pigs",
        "donuts stolen", "mermaid spotted", "zombie outbreak",
        "emotional support clown", "hiring absurdity", "company layoffs joke",
        "government conspiracy", "miracle cure", "magic pill", "celebrity cloned",
        "UFO", "time travel", "immortality pill", "giant squid", "teleportation"
    ]
    halluc_flag = any(k in text.lower() for k in hallucination_keywords)

    if halluc_flag:
        label = "FAKE"
        score = max(score, 0.95)

    # SOURCE VALIDATION - NEW FEATURE
    source_reputation = None
    source_warnings = []
    if source_url and source_url.strip():
        rep_level, emoji, description = check_source_reputation(source_url)
        source_reputation = {
            "level": rep_level,
            "emoji": emoji,
            "description": description
        }
        
        # Adjust score based on source reputation
        source_fake_score = get_source_score(rep_level)
        
        # If source is unreliable or satire, increase fake probability
        if rep_level in ["Unreliable", "Satire"]:
            label = "FAKE"
            score = max(score, 0.92)
        elif rep_level == "Highly Reliable" and label == "REAL":
            score = max(score, 0.85)  # Boost confidence for reliable sources
        
        # Check for URL warnings
        source_warnings = analyze_url_characteristics(source_url)

    # LLM fact-check
    llm_label = fact_check(text)
    if llm_label == "FAKE":
        label = "FAKE"
        score = max(score, 0.9)

    return label, score, halluc_flag, source_reputation, source_warnings

# ---------- CSS ----------
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Fake News Detector", layout="wide")

# ---------- SESSION STATE ----------
if "user_input" not in st.session_state: st.session_state.user_input = ""
if "source_url" not in st.session_state: st.session_state.source_url = ""
if "history" not in st.session_state: st.session_state.history = []
if "corrections" not in st.session_state: st.session_state.corrections = {}

# ---------- HEADER ----------
st.markdown("<h1 style='text-align:center; color:#333;'>Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#333;'>AI + Hallucination Keywords + LLM Fact-Check + Source Validation</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------- INPUT FORM ----------
with st.form(key="news_form"):
    st.subheader("Enter News Text")
    user_input = st.text_area(
        "Paste your news article here:",
        height=180,
        value=st.session_state.user_input
    )
    
    # NEW: Source URL Input
    source_url = st.text_input(
        "Source URL (optional):",
        value=st.session_state.source_url,
        placeholder="https://example.com/article",
        help="Add the URL to check source credibility"
    )

    col1, col2 = st.columns([1,1])
    with col1:
        reset = st.form_submit_button("Reset Input")
    with col2:
        submit = st.form_submit_button("Run Classification")

# ---------- FORM ACTIONS ----------
if reset:
    st.session_state.user_input = ""
    st.session_state.source_url = ""

if submit and user_input.strip():
    st.session_state.user_input = user_input
    st.session_state.source_url = source_url
    
    # Input validation
    if len(user_input.strip()) < 20:
        st.warning("⚠️ Please enter at least 20 characters for accurate classification.")
    else:
        with st.spinner("Analyzing news..."):
            try:
                classifier = load_classifier()

                if user_input in st.session_state.corrections:
                    label = st.session_state.corrections[user_input]
                    score = 1.0
                    halluc_flag = False
                    source_reputation = None
                    source_warnings = []
                else:
                    label, score, halluc_flag, source_reputation, source_warnings = hybrid_classify(
                        user_input, classifier, source_url
                    )

                st.session_state.history.insert(0, {
                    "News": user_input,
                    "Source URL": source_url,
                    "Label": label,
                    "Confidence": f"{score*100:.2f}%",
                    "Hallucination": halluc_flag,
                    "Source Reputation": source_reputation,
                    "Source Warnings": source_warnings
                })
            except Exception as e:
                st.error(f"❌ Classification error: {str(e)}")

# ---------- DISPLAY RESULTS ----------
if st.session_state.history:
    st.subheader("Results / Session History")
    
    for i, item in enumerate(st.session_state.history):
        color_class = "fake" if item["Label"]=="FAKE" else "real"
        
        with st.expander(f"{item['Label']} | Confidence: {item['Confidence']}", expanded=(i==0)):
            st.markdown(f"<div class='card {color_class} card-content'>", unsafe_allow_html=True)
            
            st.write(f"**News:** {item['News']}")
            st.write(f"**Hallucination Flag:** {item['Hallucination']}")
            
            # Display source reputation
            if item.get("Source Reputation"):
                rep = item["Source Reputation"]
                st.markdown(f"### {rep['emoji']} Source Credibility: {rep['level']}")
                st.info(f"ℹ️ {rep['description']}")
                
                # Display warnings
                if item.get("Source Warnings"):
                    st.warning("⚠️ **URL Warnings:** " + ", ".join(item["Source Warnings"]))
            elif item.get("Source URL") and item["Source URL"].strip():
                st.write(f"**Source:** {item['Source URL']}")

            # Editable label
            new_label = st.selectbox(
                f"Predicted Label ({item['Label']})",
                ["FAKE", "REAL"],
                index=0 if item['Label']=="FAKE" else 1,
                key=f"edit_label_{i}"
            )
            if st.button("Save Correction", key=f"save_{i}"):
                st.session_state.history[i]["Label"] = new_label
                st.session_state.corrections[item["News"]] = new_label
                st.success(f"✅ Label updated to {new_label}")

            # Confidence bar
            confidence_value = float(item["Confidence"].replace("%",""))
            bar_class = "confidence-fake" if item["Label"]=="FAKE" else "confidence-real"
            st.markdown(f"""
            <div class='confidence-container'>
                <div class='confidence-fill {bar_class}' style='width:{confidence_value}%'>{item["Confidence"]}</div>
            </div>
            """, unsafe_allow_html=True)

            # FAKE explanation
            if item["Label"]=="FAKE":
                with st.spinner("Generating explanation..."):
                    try:
                        explanation = explain_fake_news(item["News"])
                        st.markdown("#### 🧠 Why This May Be Fake")
                        st.write(explanation)
                    except Exception as e:
                        st.error(f"Could not generate explanation: {str(e)}")

            st.markdown("</div>", unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("""
<div class='footer'>
<hr>
Built with <strong>Streamlit</strong>, <strong>HuggingFace Transformers</strong>, and <strong>Groq LLaMA</strong>.<br>
<strong>How it works:</strong><br>
1) Base AI model predicts FAKE/REAL news from training data.<br>
2) Expanded hallucination keywords flag absurd, humorous, or sensational claims (forces FAKE).<br>
3) <strong>NEW: Source validation checks URL credibility and domain reputation.</strong><br>
4) LLM fact-check ensemble double-checks every news item (even high-confidence predictions).<br>
5) Users can reset input or manually correct labels.<br>
6) Corrected labels are remembered during the session (self-learning simulation).<br>
7) FAKE news gets a detailed AI-generated explanation.<br>
8) Confidence score visualized with a colored bar for certainty.<br>
</div>
""", unsafe_allow_html=True)