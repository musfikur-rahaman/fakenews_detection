import streamlit as st
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

# ---------- ENSEMBLE MODEL CONFIGURATION ----------
MODEL_CONFIG = {
    "primary": "mrm8488/bert-tiny-finetuned-fake-news-detection",  # Fast & accurate
    "fallback": "distilbert-base-uncased-finetuned-sst-2-english"  # General sentiment
}

@st.cache_resource(show_spinner=False)
def load_ensemble_models():
    """Load CPU-friendly ensemble models for fake news detection"""
    from transformers import pipeline

    models = {}
    model_weights = {}

    MODEL_CONFIG = {
        "primary": "mrm8488/bert-tiny-finetuned-fake-news-detection",
        "fallback": "distilbert-base-uncased-finetuned-sst-2-english"
    }

    # Primary fake news model
    try:
        models["primary"] = pipeline(
            "text-classification",
            model=MODEL_CONFIG["primary"],
            device=-1,       # CPU-only
            truncation=True,
            max_length=256   # smaller length for cloud stability
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
            device=-1,       # CPU-only
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
    
    # Handle sentiment model differently
    if model_name == "fallback":
        if "NEGATIVE" in label_str or "LABEL_0" in label_str:
            return "FAKE"  # Negative sentiment often correlates with fake news
        elif "POSITIVE" in label_str or "LABEL_1" in label_str:
            return "REAL"
        else:
            return "REAL"
    
    # Standard mappings for fake news models
    if "FAKE" in label_str or "LABEL_1" in label_str:
        return "FAKE"
    elif "REAL" in label_str or "LABEL_0" in label_str:
        return "REAL"
    else:
        return "REAL"  # Default to real to avoid false positives

# ---------- ENSEMBLE CLASSIFICATION ----------
def ensemble_classify(text, models, model_weights, source_url=None):
    """True ensemble classification with weighted voting"""
    if not models:
        return "REAL", 0.5, False, None, []
    
    predictions = []
    confidence_scores = []
    model_details = []
    
    # Get predictions from all available models
    for model_name, model in models.items():
        try:
            result = model(text[:1000])[0]  # Limit text length for stability
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
    
    # Weighted voting based on model reliability
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
    
    # Normalize scores
    if total_weight > 0:
        fake_score /= total_weight
        real_score /= total_weight
    
    # Determine final label and confidence
    if fake_score > real_score:
        final_label = "FAKE"
        final_confidence = fake_score
    else:
        final_label = "REAL" 
        final_confidence = real_score
    
    # Enhanced pattern detection
    halluc_flag = detect_hallucination_patterns(text)
    
    # Source reputation (informational only - mild influence)
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
    
    # Final fusion with heuristics (mild adjustments)
    final_label, final_confidence = fuse_predictions(
        final_label, final_confidence, halluc_flag, source_reputation, model_details
    )
    
    return final_label, final_confidence, halluc_flag, source_reputation, source_warnings, model_details

def fuse_predictions(ensemble_label, ensemble_confidence, halluc_flag, source_reputation, model_details):
    """Fuse ensemble predictions with mild heuristic adjustments"""
    label = ensemble_label
    confidence = ensemble_confidence
    
    # Mild hallucination adjustment
    if halluc_flag:
        confidence = min(confidence + 0.10, 0.90)  # Reduced from 0.15
        if confidence > 0.65 and label == "REAL":  # Only switch if not strongly real
            label = "FAKE"
    
    # Mild source reputation adjustment
    if source_reputation:
        rep_level = source_reputation.get("level", "")
        source_weight = 0.15  # Reduced influence
        
        if rep_level in ["Unreliable", "Satire"] and label == "FAKE":
            confidence = min(confidence + (0.2 * source_weight), 0.95)
        elif rep_level == "Highly Reliable" and label == "FAKE" and confidence < 0.75:
            # Give benefit of doubt to reliable sources for borderline cases
            confidence = confidence * (1 - source_weight)
    
    # Confidence calibration
    confidence = max(0.1, min(0.99, confidence))
    
    return label, confidence

def detect_hallucination_patterns(text):
    """Enhanced pattern detection with context awareness"""
    text_lower = text.lower()
    
    # Strong fake indicators (absurd claims)
    strong_indicators = [
        "microchip in vaccine", "5g caused", "flat earth", "alien body found",
        "celebrity cloned", "time travel", "flying pigs", "talking animals",
        "zombie outbreak", "immortality pill", "magic cure", "overnight millionaire",
        "government hiding aliens", "secret cancer cure", "world ending tomorrow",
        "emotional support clown", "hiring absurdity", "donuts stolen by squirrels"
    ]
    
    # Check for strong indicators
    for indicator in strong_indicators:
        if indicator in text_lower:
            return True
    
    # Contextual indicators (need multiple to trigger)
    contextual_indicators = [
        "breaking news", "shocking discovery", "they don't want you to know",
        "doctors hate this", "miracle cure", "secret revealed", "cover-up",
        "leaked documents", "forbidden knowledge", "mainstream media won't tell you"
    ]
    
    context_count = sum(1 for indicator in contextual_indicators if indicator in text_lower)
    return context_count >= 3  # Require multiple contextual indicators

# ---------- CSS ----------
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Fake News Detector", layout="wide", page_icon="")

# ---------- SESSION STATE ----------
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

# ---------- HEADER ----------
st.markdown("<h1 style='text-align:center; color:#333;'>Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#666; margin-bottom:10px;'>Multiple AI Models + Smart Source Analysis + Balanced Classification</div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#333; font-size:14px;'>Ensemble approach for consistent & accurate results</div>", unsafe_allow_html=True)
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
            # Fallback to single model
            try:
                st.session_state.ensemble_models = {
                    "primary": pipeline("text-classification", model=MODEL_CONFIG["primary"])
                }
                st.session_state.model_weights = {"primary": 1.0}
                st.session_state.models_loaded = True
            except Exception as e:
                st.error(f"❌ Fallback model also failed: {e}")

# ---------- INSTRUCTIONS ----------
with st.expander("How to Use", expanded=False):
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
    st.subheader("Enter News Text or URL")
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
    
    # Input validation
    if len(user_input.strip()) < 10:
        st.warning("⚠️ Please enter at least 10 characters.")
    else:
        # Normalize input
        normalized_input = normalize_url(user_input)
        
        # Detect if input is URL or text
        if is_url(normalized_input):
            st.info("🔗 URL detected! Fetching article content...")
            
            with st.spinner("📡 Fetching article from URL..."):
                article_text, article_title, error = extract_article_content(normalized_input)
            
            if error:
                st.error(error)
                st.info("💡 Tip: Make sure the URL is accessible and contains an article.")
            else:
                st.success(f"✅ Article fetched: **{article_title}**")
                
                # Show preview
                with st.expander("👁️ Preview Extracted Content", expanded=False):
                    st.write(f"**Title:** {article_title}")
                    st.write(f"**Content Length:** {len(article_text)} characters")
                    st.write(f"**Preview:** {article_text[:500]}...")
                
                # Use extracted content for classification
                text_to_analyze = article_text
                source_url = normalized_input
        else:
            st.info("📄 Text detected! Analyzing content...")
            text_to_analyze = user_input
            source_url = None
        
        # Perform ensemble classification
        if 'text_to_analyze' in locals() and st.session_state.models_loaded:
            with st.spinner("🤖 Running ensemble analysis with multiple AI models..."):
                try:
                    # Check for manual corrections
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

                    # Store result
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
    
    # Statistics
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
            
            # Show article title if from URL
            if item.get("Article Title"):
                st.markdown(f"### 📰 {item['Article Title']}")
            
            # Show content preview
            news_preview = item['News'][:300] + "..." if len(item['News']) > 300 else item['News']
            st.write(f"**Content Preview:** {news_preview}")
            
            # Show full content in expander
            if len(item['News']) > 300:
                with st.expander("📖 Show Full Content"):
                    st.write(item['News'])
            
            # Ensemble model details
            st.markdown("#### 🤖 Ensemble Model Results")
            if item.get("Model Details"):
                for model_info in item["Model Details"]:
                    model_emoji = "❌" if model_info["label"] == "FAKE" else "✅"
                    st.write(f"{model_emoji} **{model_info['model'].title()}**: {model_info['label']} ({model_info['confidence']:.1%})")
            else:
                st.write("Single model analysis")
            
            st.write(f"**Suspicious Patterns:** {'Yes ⚠️' if item['Hallucination'] else 'No ✅'}")
            
            # Display source reputation
            if item.get("Source Reputation"):
                rep = item["Source Reputation"]
                st.markdown(f"### {rep['emoji']} Source Analysis")
                st.info(f"**{rep['level']}**: {rep['description']}")
                
                if item.get("Source URL"):
                    domain = extract_domain(item["Source URL"])
                    st.write(f"**Domain:** {domain}")
                
                # Display warnings
                if item.get("Source Warnings"):
                    st.warning("🚨 **URL Analysis:** " + ", ".join(item["Source Warnings"]))

            # Analysis details
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

            # Confidence bar
            confidence_value = float(item["Confidence"].replace("%",""))
            bar_class = "confidence-fake" if item["Label"]=="FAKE" else "confidence-real"
            st.markdown(f"""
            <div class='confidence-container'>
                <div class='confidence-fill {bar_class}' style='width:{confidence_value}%'>{item["Confidence"]}</div>
            </div>
            """, unsafe_allow_html=True)

            # Manual correction
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

            # FAKE explanation
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

Built with <strong>Streamlit</strong> • <strong>Ensemble AI Models</strong> • <strong>Balanced Detection</strong><br>

<strong>🔬 Pipeline:</strong> Smart input detection (URL vs Text), Automatic article extraction, Multiple AI model ensemble voting, Balanced credibility analysis, Pattern detection, LLM fact-checking, Confidence calibration, Manual correction & learning.<br>

<strong>🤖 Models:</strong> mrm8488/bert-tiny-finetuned-fake-news-detection, DistilBERT Sentiment, LLaMA 4 Scout.

</div>
""", unsafe_allow_html=True)
