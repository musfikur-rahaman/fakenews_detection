import streamlit as st
from supabase import create_client, Client
from transformers import pipeline
from llmhelper import explain_fake_news
from source_validator import (
    check_source_reputation, 
    get_source_score, 
    analyze_url_characteristics,
    extract_domain
)
from url_content_fetcher import is_url, extract_article_content, normalize_url
import re   
import time
import numpy as np
import os
from dotenv import load_dotenv
#from supabase_py import create_client, Client


# ---------- LOAD ENV VARIABLES ----------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase URL or Key not found. Please set them in .env file.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------- PAGE SETUP (Move to top) ----------
st.set_page_config(page_title="Fake News Detector", layout="wide", page_icon="🔍")

# ---------- CSS ----------
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ----------------- SESSION STATE -----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "user_input" not in st.session_state: 
    st.session_state.user_input = ""
if "history" not in st.session_state: 
    st.session_state.history = []
if "corrections" not in st.session_state: 
    st.session_state.corrections = {}
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "ensemble_models" not in st.session_state:
    st.session_state.ensemble_models = None
if "model_weights" not in st.session_state:
    st.session_state.model_weights = None

# ----------------- SUPABASE AUTH FUNCTIONS -----------------
def login(email, password):
    """Authenticate user with Supabase"""
    try:
        # Sign in with Supabase Auth
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        if response.user:
            st.session_state.logged_in = True
            st.session_state.user_email = response.user.email
            st.session_state.user_id = response.user.id
            st.success(f"✅ Welcome back, {response.user.email}!")
            time.sleep(1)
            st.rerun()
        else:
            st.error("❌ Login failed. Please check your credentials.")
            
    except Exception as e:
        error_msg = str(e)
        if "Invalid login credentials" in error_msg:
            st.error("❌ Invalid email or password")
        elif "Email not confirmed" in error_msg:
            st.error("❌ Please confirm your email before logging in")
        else:
            st.error(f"❌ Login error: {error_msg}")

def signup(email, password):
    """Register new user with Supabase"""
    try:
        # Sign up with Supabase Auth
        response = supabase.auth.sign_up({
            "email": email,
            "password": password
        })
        
        if response.user:
            st.success("✅ Account created successfully!")
            
            # Check if email confirmation is required
            if response.session:
                # Auto-login if no confirmation needed
                st.session_state.logged_in = True
                st.session_state.user_email = response.user.email
                st.session_state.user_id = response.user.id
                st.info("🎉 You're now logged in!")
                time.sleep(1)
                st.rerun()
            else:
                # Email confirmation required
                st.info("📧 Please check your email to confirm your account, then login.")
        else:
            st.error("❌ Signup failed. Please try again.")
            
    except Exception as e:
        error_msg = str(e)
        if "User already registered" in error_msg:
            st.error("❌ This email is already registered. Please login instead.")
        elif "Password should be at least" in error_msg:
            st.error("❌ Password must be at least 6 characters long")
        elif "Unable to validate email address" in error_msg:
            st.error("❌ Please enter a valid email address")
        else:
            st.error(f"❌ Signup error: {error_msg}")

def logout():
    """Logout user from Supabase"""
    try:
        supabase.auth.sign_out()
        st.session_state.logged_in = False
        st.session_state.user_email = None
        st.session_state.user_id = None
        st.session_state.models_loaded = False
        st.session_state.ensemble_models = None
        st.session_state.history = []
        st.success("✅ Logged out successfully!")
        time.sleep(1)
        st.rerun()
    except Exception as e:
        st.error(f"❌ Logout error: {str(e)}")

# Check for existing session on page load
def check_existing_session():
    """Check if user has an active Supabase session"""
    try:
        session = supabase.auth.get_session()
        if session and session.user:
            st.session_state.logged_in = True
            st.session_state.user_email = session.user.email
            st.session_state.user_id = session.user.id
            return True
    except:
        pass
    return False

# Check for existing session
if not st.session_state.logged_in:
    check_existing_session()

# ----------------- LOGIN PAGE -----------------
if not st.session_state.logged_in:
    st.title("🔍 Welcome to Fake News Detector")
    st.write("Detect fake news with AI-powered analysis")
    st.markdown("---")

    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        st.subheader("Login to Your Account")
        with st.form(key="login_form"):
            login_email = st.text_input("Email", key="login_email", placeholder="your@email.com")
            login_password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
            login_button = st.form_submit_button("🔐 Login", use_container_width=True, type="primary")
            
            if login_button:
                if login_email and login_password:
                    with st.spinner("Authenticating..."):
                        login(login_email, login_password)
                else:
                    st.error("⚠️ Please enter both email and password")

    with tab2:
        st.subheader("Create New Account")
        with st.form(key="signup_form"):
            signup_email = st.text_input("Email", key="signup_email", placeholder="your@email.com")
            signup_password = st.text_input("Password", type="password", key="signup_password", placeholder="At least 6 characters")
            signup_confirm = st.text_input("Confirm Password", type="password", key="signup_confirm", placeholder="Re-enter password")
            
            st.caption("Password must be at least 6 characters long")
            signup_button = st.form_submit_button("✍️ Create Account", use_container_width=True, type="primary")
            
            if signup_button:
                if not signup_email or not signup_password or not signup_confirm:
                    st.error("⚠️ Please fill in all fields")
                elif len(signup_password) < 6:
                    st.error("⚠️ Password must be at least 6 characters long")
                elif signup_password != signup_confirm:
                    st.error("⚠️ Passwords do not match")
                else:
                    with st.spinner("Creating account..."):
                        signup(signup_email, signup_password)
    
    # Footer for login page
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
    🔒 Secured with Supabase Authentication<br>
    Your data is protected and encrypted
    </div>
    """, unsafe_allow_html=True)
    
    st.stop()  # Critical: Stop execution here to prevent main app from loading

# ----------------- MAIN APP (Only loads if logged in) -----------------

# ---------- ENSEMBLE MODEL CONFIGURATION ----------
MODEL_CONFIG = {
    "primary": "mrm8488/bert-tiny-finetuned-fake-news-detection",
    "fallback": "distilbert-base-uncased-finetuned-sst-2-english"
}

@st.cache_resource(show_spinner=False)
def load_ensemble_models():
    """Load CPU-friendly ensemble models for fake news detection"""
    models = {}
    model_weights = {}

    # Primary fake news model
    try:
        models["primary"] = pipeline(
            "text-classification",
            model=MODEL_CONFIG["primary"],
            device=-1,
            truncation=True,
            max_length=256
        )
        model_weights["primary"] = 0.7
        st.success("✅ Loaded primary fake news model")
    except Exception as e:
        st.error(f"❌ Primary model failed: {e}")

    # Fallback sentiment model
    try:
        models["fallback"] = pipeline(
            "text-classification",
            model=MODEL_CONFIG["fallback"],
            device=-1,
            truncation=True,
            max_length=256
        )
        model_weights["fallback"] = 0.3
        st.success("✅ Loaded fallback sentiment model")
    except Exception as e:
        st.warning(f"⚠️ Fallback model failed: {e}")

    return models, model_weights

# ---------- LABEL MAPPING ----------
def map_label(label, model_name="primary"):
    """Consistent label mapping across different models"""
    label_str = str(label).upper()
    
    if model_name == "fallback":
        if "NEGATIVE" in label_str or "LABEL_0" in label_str:
            return "FAKE"
        elif "POSITIVE" in label_str or "LABEL_1" in label_str:
            return "REAL"
        else:
            return "REAL"
    
    if "FAKE" in label_str or "LABEL_1" in label_str:
        return "FAKE"
    elif "REAL" in label_str or "LABEL_0" in label_str:
        return "REAL"
    else:
        return "REAL"

# ---------- ENSEMBLE CLASSIFICATION ----------
def ensemble_classify(text, models, model_weights, source_url=None):
    """True ensemble classification with weighted voting"""
    if not models:
        return "REAL", 0.5, False, None, []
    
    predictions = []
    confidence_scores = []
    model_details = []
    
    for model_name, model in models.items():
        try:
            result = model(text[:1000])[0]
            label = map_label(result['label'], model_name)
            score = result['score']
            
            predictions.append(label)
            confidence_scores.append(score)
            model_details.append({
                "model": model_name,
                "label": label,
                "confidence": score,
                "weight": model_weights.get(model_name, 0.1)
            })
            
        except Exception as e:
            st.warning(f"Model {model_name} failed: {e}")
            continue
    
    if not predictions:
        st.warning("⚠️ All models failed, using fallback classification")
        return "REAL", 0.5, False, None, []
    
    fake_score = 0
    real_score = 0
    total_weight = 0
    
    for i, (model_name, prediction) in enumerate(zip(models.keys(), predictions)):
        if model_name in model_weights:
            weight = model_weights[model_name]
            confidence = confidence_scores[i]
            
            if prediction == "FAKE":
                fake_score += weight * confidence
            else:
                real_score += weight * confidence
            
            total_weight += weight
    
    if total_weight > 0:
        fake_score /= total_weight
        real_score /= total_weight
    
    if fake_score > real_score:
        final_label = "FAKE"
        final_confidence = fake_score
    else:
        final_label = "REAL" 
        final_confidence = real_score
    
    halluc_flag = detect_hallucination_patterns(text)
    
    source_reputation = None
    source_warnings = []
    if source_url and source_url.strip():
        rep_level, emoji, description = check_source_reputation(source_url)
        source_reputation = {
            "level": rep_level,
            "emoji": emoji,
            "description": description
        }
        source_warnings = analyze_url_characteristics(source_url)
    
    final_label, final_confidence = fuse_predictions(
        final_label, final_confidence, halluc_flag, source_reputation, model_details
    )
    
    return final_label, final_confidence, halluc_flag, source_reputation, source_warnings, model_details

def fuse_predictions(ensemble_label, ensemble_confidence, halluc_flag, source_reputation, model_details):
    """Fuse ensemble predictions with mild heuristic adjustments"""
    label = ensemble_label
    confidence = ensemble_confidence
    
    if halluc_flag:
        confidence = min(confidence + 0.10, 0.90)
        if confidence > 0.65 and label == "REAL":
            label = "FAKE"
    
    if source_reputation:
        rep_level = source_reputation.get("level", "")
        source_weight = 0.15
        
        if rep_level in ["Unreliable", "Satire"] and label == "FAKE":
            confidence = min(confidence + (0.2 * source_weight), 0.95)
        elif rep_level == "Highly Reliable" and label == "FAKE" and confidence < 0.75:
            confidence = confidence * (1 - source_weight)
    
    confidence = max(0.1, min(0.99, confidence))
    
    return label, confidence

def detect_hallucination_patterns(text):
    """Enhanced pattern detection with context awareness"""
    text_lower = text.lower()
    
    strong_indicators = [
        "microchip in vaccine", "5g caused", "flat earth", "alien body found",
        "celebrity cloned", "time travel", "flying pigs", "talking animals",
        "zombie outbreak", "immortality pill", "magic cure", "overnight millionaire",
        "government hiding aliens", "secret cancer cure", "world ending tomorrow",
        "emotional support clown", "hiring absurdity", "donuts stolen by squirrels"
    ]
    
    for indicator in strong_indicators:
        if indicator in text_lower:
            return True
    
    contextual_indicators = [
        "breaking news", "shocking discovery", "they don't want you to know",
        "doctors hate this", "miracle cure", "secret revealed", "cover-up",
        "leaked documents", "forbidden knowledge", "mainstream media won't tell you"
    ]
    
    context_count = sum(1 for indicator in contextual_indicators if indicator in text_lower)
    return context_count >= 3

# ---------- HEADER ----------
st.markdown("<h1 style='text-align:center; color:#333;'>🔍 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown(f"<div style='text-align:center; color:#666; margin-bottom:10px;'>Welcome, {st.session_state.user_email}!</div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#333; font-size:14px;'>Multiple AI Models + Smart Source Analysis + Balanced Classification</div>", unsafe_allow_html=True)

# Logout button
col1, col2, col3 = st.columns([4, 1, 4])
with col2:
    if st.button("🚪 Logout", use_container_width=True):
        logout()

st.markdown("---")

# ---------- MODEL LOADING ----------
if not st.session_state.models_loaded:
    with st.spinner("🔄 Loading ensemble AI models... This may take a moment."):
        models, weights = load_ensemble_models()
        if models:
            st.session_state.ensemble_models = models
            st.session_state.model_weights = weights
            st.session_state.models_loaded = True
            st.success(f"✅ Loaded {len(models)} models for ensemble classification!")
        else:
            st.error("❌ Failed to load ensemble models. Using single model fallback.")
            try:
                st.session_state.ensemble_models = {
                    "primary": pipeline("text-classification", model=MODEL_CONFIG["primary"])
                }
                st.session_state.model_weights = {"primary": 1.0}
                st.session_state.models_loaded = True
            except Exception as e:
                st.error(f"❌ Fallback model also failed: {e}")

# ---------- INSTRUCTIONS ----------
with st.expander("ℹ️ How to Use", expanded=False):
    st.markdown("""
    **Enter either:**
    - **News text** directly in the box
    - **URL** of a news article (e.g., `https://bbc.com/news/article`)
    
    **Ensemble Analysis Pipeline:**
    - Smart input detection (URL vs Text) with automatic article extraction
    - Multiple AI models working together using weighted voting
    - Source credibility, advanced pattern detection, and LLM fact-checking
    - Balanced and reliable final classification
    
    **Models in Ensemble:**
    - Primary: Fake news detection (70% weight)
    - Fallback: Sentiment analysis (30% weight)
    """)

# ---------- INPUT FORM ----------
with st.form(key="news_form"):
    st.subheader("📝 Enter News Text or URL")
    user_input = st.text_area(
        "Paste news article or enter URL:",
        height=180,
        value=st.session_state.user_input,
        placeholder="Enter news text OR paste a URL like https://example.com/article",
        help="Smart input: detects URLs automatically and fetches content"
    )

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        reset = st.form_submit_button("🔄 Reset", use_container_width=True)
    with col2:
        submit = st.form_submit_button("🚀 Analyze", use_container_width=True, type="primary")
    with col3:
        clear_history = st.form_submit_button("🗑️ Clear History", use_container_width=True)

# ---------- FORM ACTIONS ----------
if reset:
    st.session_state.user_input = ""
    st.rerun()

if clear_history:
    st.session_state.history = []
    st.session_state.corrections = {}
    st.success("✅ History cleared!")
    st.rerun()

if submit and user_input.strip():
    st.session_state.user_input = user_input
    
    if len(user_input.strip()) < 10:
        st.warning("⚠️ Please enter at least 10 characters.")
    else:
        normalized_input = normalize_url(user_input)
        
        if is_url(normalized_input):
            st.info("🔗 URL detected! Fetching article content...")
            
            with st.spinner("📡 Fetching article from URL..."):
                article_text, article_title, error = extract_article_content(normalized_input)
            
            if error:
                st.error(error)
                st.info("💡 Tip: Make sure the URL is accessible and contains an article.")
            else:
                st.success(f"✅ Article fetched: **{article_title}**")
                
                with st.expander("👁️ Preview Extracted Content", expanded=False):
                    st.write(f"**Title:** {article_title}")
                    st.write(f"**Content Length:** {len(article_text)} characters")
                    st.write(f"**Preview:** {article_text[:500]}...")
                
                text_to_analyze = article_text
                source_url = normalized_input
        else:
            st.info("📄 Text detected! Analyzing content...")
            text_to_analyze = user_input
            source_url = None
        
        if 'text_to_analyze' in locals() and st.session_state.models_loaded:
            with st.spinner("🤖 Running ensemble analysis with multiple AI models..."):
                try:
                    if text_to_analyze in st.session_state.corrections:
                        label = st.session_state.corrections[text_to_analyze]
                        score = 1.0
                        halluc_flag = False
                        source_reputation = None
                        source_warnings = []
                        model_details = []
                    else:
                        label, score, halluc_flag, source_reputation, source_warnings, model_details = ensemble_classify(
                            text_to_analyze, 
                            st.session_state.ensemble_models, 
                            st.session_state.model_weights,
                            source_url
                        )

                    st.session_state.history.insert(0, {
                        "News": text_to_analyze,
                        "Original Input": user_input,
                        "Source URL": source_url,
                        "Article Title": article_title if 'article_title' in locals() else None,
                        "Label": label,
                        "Confidence": f"{score*100:.2f}%",
                        "Hallucination": halluc_flag,
                        "Source Reputation": source_reputation,
                        "Source Warnings": source_warnings,
                        "Model Details": model_details,
                        "Timestamp": time.time()
                    })
                    
                    st.success("✅ Ensemble analysis complete!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Classification error: {str(e)}")
        else:
            st.error("❌ AI models not loaded. Please refresh the page.")

# ---------- DISPLAY RESULTS ----------
if st.session_state.history:
    st.markdown("---")
    st.subheader("📊 Results & History")
    
    total = len(st.session_state.history)
    fake_count = sum(1 for item in st.session_state.history if item["Label"] == "FAKE")
    real_count = total - fake_count
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Analyzed", total)
    col2.metric("Fake News", fake_count, delta=f"{fake_count/total*100:.1f}%" if total > 0 else "0%")
    col3.metric("Real News", real_count, delta=f"{real_count/total*100:.1f}%" if total > 0 else "0%")
    col4.metric("Accuracy", f"{(real_count/total*100):.1f}%" if total > 0 else "0%")
    
    st.markdown("---")
    
    for i, item in enumerate(st.session_state.history):
        color_class = "fake" if item["Label"]=="FAKE" else "real"
        emoji = "❌" if item["Label"]=="FAKE" else "✅"
        
        with st.expander(f"{emoji} {item['Label']} | Confidence: {item['Confidence']} | Ensemble", expanded=(i==0)):
            st.markdown(f"<div class='card {color_class} card-content'>", unsafe_allow_html=True)
            
            if item.get("Article Title"):
                st.markdown(f"### 📰 {item['Article Title']}")
            
            news_preview = item['News'][:300] + "..." if len(item['News']) > 300 else item['News']
            st.write(f"**Content Preview:** {news_preview}")
            
            if len(item['News']) > 300:
                with st.expander("📖 Show Full Content"):
                    st.write(item['News'])
            
            st.markdown("#### 🤖 Ensemble Model Results")
            if item.get("Model Details"):
                for model_info in item["Model Details"]:
                    model_emoji = "❌" if model_info["label"] == "FAKE" else "✅"
                    st.write(f"{model_emoji} **{model_info['model'].title()}**: {model_info['label']} ({model_info['confidence']:.1%})")
            else:
                st.write("Single model analysis")
            
            st.write(f"**Suspicious Patterns:** {'Yes ⚠️' if item['Hallucination'] else 'No ✅'}")
            
            if item.get("Source Reputation"):
                rep = item["Source Reputation"]
                st.markdown(f"### {rep['emoji']} Source Analysis")
                st.info(f"**{rep['level']}**: {rep['description']}")
                
                if item.get("Source URL"):
                    domain = extract_domain(item["Source URL"])
                    st.write(f"**Domain:** {domain}")
                
                if item.get("Source Warnings"):
                    st.warning("🚨 **URL Analysis:** " + ", ".join(item["Source Warnings"]))

            st.markdown("#### 🔍 Analysis Details")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Final Confidence:** {item['Confidence']}")
                st.write(f"**Pattern Detection:** {'Triggered' if item['Hallucination'] else 'Clear'}")
            with col2:
                if item.get("Source Reputation"):
                    st.write(f"**Source Influence:** Mild")
                else:
                    st.write(f"**Source Influence:** None")
                st.write(f"**Models Used:** {len(item.get('Model Details', []))}")

            confidence_value = float(item["Confidence"].replace("%",""))
            bar_class = "confidence-fake" if item["Label"]=="FAKE" else "confidence-real"
            st.markdown(f"""
            <div class='confidence-container'>
                <div class='confidence-fill {bar_class}' style='width:{confidence_value}%'>{item["Confidence"]}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### ✏️ Manual Verification")
            col1, col2 = st.columns([2, 1])
            with col1:
                new_label = st.selectbox(
                    "Correct classification if needed:",
                    ["FAKE", "REAL"],
                    index=0 if item['Label']=="FAKE" else 1,
                    key=f"edit_label_{i}"
                )
            with col2:
                if st.button("💾 Save Correction", key=f"save_{i}", use_container_width=True):
                    st.session_state.history[i]["Label"] = new_label
                    st.session_state.corrections[item["News"]] = new_label
                    st.success(f"✅ Updated to {new_label}")
                    st.rerun()

            if item["Label"]=="FAKE":
                with st.spinner("🧠 Generating detailed explanation..."):
                    try:
                        explanation = explain_fake_news(item["News"][:800])
                        st.markdown("#### 🧠 Why This May Be Fake")
                        st.info(explanation)
                    except Exception as e:
                        st.error(f"Could not generate explanation: {str(e)}")

            st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div class='footer' style='font-size: 0.9em; line-height: 1.3em;'>
Built with <strong>Streamlit</strong> • <strong>Ensemble AI Models</strong> • <strong>Supabase Auth</strong><br>
<strong>🔬 Pipeline:</strong> Smart input detection (URL vs Text), Automatic article extraction, Multiple AI model ensemble voting, Balanced credibility analysis, Pattern detection, LLM fact-checking, Confidence calibration, Manual correction & learning.<br>
<strong>🤖 Models:</strong> mrm8488/bert-tiny-finetuned-fake-news-detection, DistilBERT Sentiment, LLaMA 4 Scout.
</div>
""", unsafe_allow_html=True)