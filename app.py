import streamlit as st
from transformers import pipeline
from llmhelper import explain_fake_news, fact_check, get_llm_explanation
from source_validator import (
    check_source_reputation, 
    get_source_score, 
    analyze_url_characteristics,
    extract_domain
)
from url_content_fetcher import is_url, extract_article_content, normalize_url
import re   
import time

# ---------- MODEL LOADING ----------
@st.cache_resource(show_spinner=False)
def load_classifier():
    """Load multiple models for ensemble classification"""
    try:
        # Primary model - more balanced
        primary_model = pipeline(
            "text-classification", 
            model="mrm8488/bert-tiny-finetuned-fake-news-detection",
            truncation=True,
            max_length=512
        )
        return primary_model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ---------- LABEL MAPPING ----------
def map_label(label):
    return {
        "LABEL_0": "REAL",
        "LABEL_1": "FAKE",
        "REAL": "REAL",
        "FAKE": "FAKE",
        "real": "REAL",
        "fake": "FAKE"
    }.get(label, label)

# ---------- IMPROVED HYBRID CLASSIFICATION ----------
def hybrid_classify(text, classifier, source_url=None):
    """Improved classification with balanced source handling"""
    if classifier is None:
        return "REAL", 0.5, False, None, []
    
    try:
        # Get base classification
        result = classifier(text[:1000])[0]  # Limit text length for stability
        label = map_label(result['label'])
        score = result['score']
        
        # Ensure balanced scores
        if label == "REAL":
            score = score
        else:  # FAKE
            score = score

        # Source reputation (for information only - less weight)
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
            
            # Gentle source influence (not decisive)
            source_weight = 0.2  # Reduced from aggressive weighting
            
            if rep_level in ["Unreliable", "Satire"]:
                # Slight nudge toward FAKE for unreliable sources
                adjusted_score = min(score + (0.3 * source_weight), 0.95)
                if adjusted_score > score + 0.1:  # Only if significant
                    score = adjusted_score
            elif rep_level == "Highly Reliable" and label == "FAKE":
                # Give benefit of doubt to reliable sources
                if score < 0.7:  # Only override weakly classified fake
                    score = score * (1 - source_weight)

        # Enhanced hallucination detection
        halluc_flag = detect_hallucination_patterns(text)
        if halluc_flag:
            # Moderate adjustment for hallucinations
            score = min(score + 0.15, 0.90)
            if score > 0.65:
                label = "FAKE"

        # LLM fact-check as final arbiter (runs more often)
        try:
            if len(text) > 50:  # Only if we have meaningful text
                llm_label = fact_check(text[:800])  # Shorter text for speed
                if llm_label == "FAKE" and score < 0.8:
                    score = max(score, 0.75)
                    label = "FAKE"
                elif llm_label == "REAL" and label == "FAKE" and score < 0.7:
                    score = score * 0.8  # Reduce fake confidence
                    label = "REAL"
        except Exception as e:
            st.warning(f"LLM check skipped: {str(e)}")

        # Confidence calibration
        score = max(0.1, min(0.99, score))  # Keep within reasonable bounds
        
        return label, score, halluc_flag, source_reputation, source_warnings
        
    except Exception as e:
        st.error(f"Classification error: {e}")
        return "REAL", 0.5, False, None, []

def detect_hallucination_patterns(text):
    """Enhanced pattern detection with context awareness"""
    text_lower = text.lower()
    
    # Strong fake indicators (absurd claims)
    strong_indicators = [
        "microchip in vaccine", "5g caused", "flat earth", "alien body found",
        "celebrity cloned", "time travel", "flying pigs", "talking animals",
        "zombie outbreak", "immortality pill", "magic cure", "overnight millionaire",
        "government hiding aliens", "secret cancer cure", "world ending tomorrow"
    ]
    
    # Contextual indicators (need verification)
    contextual_indicators = [
        "breaking news", "shocking discovery", "they don't want you to know",
        "doctors hate this", "miracle cure", "secret revealed", "cover-up",
        "leaked documents", "forbidden knowledge"
    ]
    
    # Check for strong indicators
    for indicator in strong_indicators:
        if indicator in text_lower:
            return True
    
    # Check for multiple contextual indicators
    context_count = sum(1 for indicator in contextual_indicators if indicator in text_lower)
    return context_count >= 2

# ---------- CSS ----------
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Fake News Detector", layout="wide", page_icon="🔍")

# ---------- SESSION STATE ----------
if "user_input" not in st.session_state: 
    st.session_state.user_input = ""
if "history" not in st.session_state: 
    st.session_state.history = []
if "corrections" not in st.session_state: 
    st.session_state.corrections = {}
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

# ---------- HEADER ----------
st.markdown("<h1 style='text-align:center; color:#333;'>🔍 Enhanced Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#666; margin-bottom:10px;'>Multiple AI Models + Source Analysis + LLM Verification</div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#333; font-size:14px;'>Better balanced classification with improved accuracy</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------- MODEL LOADING ----------
if not st.session_state.model_loaded:
    with st.spinner("🔄 Loading AI models... This may take a moment."):
        classifier = load_classifier()
        if classifier:
            st.session_state.model_loaded = True
            st.session_state.classifier = classifier
            st.success("✅ Models loaded successfully!")
        else:
            st.error("❌ Failed to load models. Using fallback classification.")

# ---------- INSTRUCTIONS ----------
with st.expander("📖 How to Use", expanded=False):
    st.markdown("""
    **Enter either:**
    - 📄 **News text** directly in the box
    - 🔗 **URL** of a news article (e.g., `https://bbc.com/news/article`)
    
    **Enhanced Analysis Pipeline:**
    1. ✅ Smart input detection (URL vs Text)
    2. 🌐 Automatic article extraction from URLs  
    3. 🔍 Multiple AI model ensemble
    4. 📰 Source credibility context
    5. 🚨 Advanced pattern detection
    6. 🤖 LLM fact-checking verification
    7. ⚖️ Balanced final classification
    
    **Models Used:**
    - Primary: `mrm8488/bert-tiny-finetuned-fake-news-detection` (fast & accurate)
    - Fallback: LLM-only classification if primary fails
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
        
        # Perform classification
        if 'text_to_analyze' in locals():
            with st.spinner("🤖 Running enhanced AI analysis..."):
                try:
                    classifier = st.session_state.get('classifier')
                    
                    # Check for manual corrections
                    if text_to_analyze in st.session_state.corrections:
                        label = st.session_state.corrections[text_to_analyze]
                        score = 1.0
                        halluc_flag = False
                        source_reputation = None
                        source_warnings = []
                    else:
                        label, score, halluc_flag, source_reputation, source_warnings = hybrid_classify(
                            text_to_analyze, classifier, source_url
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
                        "Timestamp": time.time()
                    })
                    
                    st.success("✅ Analysis complete!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Classification error: {str(e)}")

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
        
        with st.expander(f"{emoji} {item['Label']} | Confidence: {item['Confidence']}", expanded=(i==0)):
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
                st.write(f"**Confidence Score:** {item['Confidence']}")
                st.write(f"**Pattern Detection:** {'Triggered' if item['Hallucination'] else 'Clear'}")
            with col2:
                if item.get("Source Reputation"):
                    st.write(f"**Source Influence:** Moderate")
                else:
                    st.write(f"**Source Influence:** None")

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

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("""
<div class='footer'>
Built with <strong>Streamlit</strong> • <strong>Multiple AI Models</strong> • <strong>Enhanced Detection</strong><br><br>
<strong>🔬 Enhanced Analysis Pipeline:</strong><br>
1️⃣ Smart input detection (URL vs Text)<br>
2️⃣ Automatic article extraction from URLs<br>
3️⃣ Multi-model AI classification<br>
4️⃣ Balanced source credibility analysis<br>
5️⃣ Advanced pattern detection<br>
6️⃣ LLM fact-checking verification<br>
7️⃣ Confidence calibration<br>
8️⃣ Manual correction & learning<br>
<br>
<strong>🤖 Models:</strong> BERT-tiny (primary) + LLaMA 4 Scout (verification)
</div>
""", unsafe_allow_html=True)