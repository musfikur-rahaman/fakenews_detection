import streamlit as st
from transformers import pipeline
from llmhelper import explain_fake_news, fact_check
from source_validator import (
    check_source_reputation, 
    get_source_score, 
    analyze_url_characteristics,
    extract_domain
)
from url_content_fetcher import is_url, extract_article_content, normalize_url
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

    # SOURCE VALIDATION
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
            score = max(score, 0.85)
        
        # Check for URL warnings
        source_warnings = analyze_url_characteristics(source_url)

    # LLM fact-check (only for non-reliable sources or low confidence)
    try:
        # Only use LLM override if base model is uncertain OR source is questionable
        if score < 0.7 or (source_reputation and source_reputation["level"] not in ["Highly Reliable", "Generally Reliable"]):
            llm_label = fact_check(text[:1000])  # Limit text for API
            if llm_label == "FAKE":
                label = "FAKE"
                score = max(score, 0.85)  # Reduced from 0.9
    except Exception as e:
        pass  # Silently fail instead of showing warning

    return label, score, halluc_flag, source_reputation, source_warnings

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

# ---------- HEADER ----------
st.markdown("<h1 style='text-align:center; color:#333;'>🔍 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#666; margin-bottom:10px;'>Paste news text OR enter a URL - we'll handle both!</div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#333; font-size:14px;'>AI + Hallucination Keywords + LLM Fact-Check + Source Validation</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------- INSTRUCTIONS ----------
with st.expander("📖 How to Use", expanded=False):
    st.markdown("""
    **Enter either:**
    - 📄 **News text** directly in the box
    - 🔗 **URL** of a news article (e.g., `https://bbc.com/news/article`)
    
    **The system will:**
    1. ✅ Auto-detect if you entered a URL or text
    2. 🌐 Fetch article content if it's a URL
    3. 🔍 Check source credibility
    4. 🤖 Analyze with AI models
    5. 📊 Show detailed results with confidence scores
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
            with st.spinner("🤖 Running AI analysis..."):
                try:
                    classifier = load_classifier()

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

                    st.session_state.history.insert(0, {
                        "News": text_to_analyze,
                        "Original Input": user_input,
                        "Source URL": source_url,
                        "Article Title": article_title if 'article_title' in locals() else None,
                        "Label": label,
                        "Confidence": f"{score*100:.2f}%",
                        "Hallucination": halluc_flag,
                        "Source Reputation": source_reputation,
                        "Source Warnings": source_warnings
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
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Analyzed", total)
    col2.metric("Fake News", fake_count, delta=f"{fake_count/total*100:.1f}%")
    col3.metric("Real News", real_count, delta=f"{real_count/total*100:.1f}%")
    
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
            
            st.write(f"**Hallucination Keywords Detected:** {'Yes ⚠️' if item['Hallucination'] else 'No'}")
            
            # Display source reputation
            if item.get("Source Reputation"):
                rep = item["Source Reputation"]
                st.markdown(f"### {rep['emoji']} Source Credibility: {rep['level']}")
                st.info(f"ℹ️ {rep['description']}")
                
                if item.get("Source URL"):
                    st.write(f"**Source URL:** {item['Source URL']}")
                
                # Display warnings
                if item.get("Source Warnings"):
                    st.warning("⚠️ **URL Warnings:** " + ", ".join(item["Source Warnings"]))

            # Editable label
            col1, col2 = st.columns([2, 1])
            with col1:
                new_label = st.selectbox(
                    "Adjust Classification:",
                    ["FAKE", "REAL"],
                    index=0 if item['Label']=="FAKE" else 1,
                    key=f"edit_label_{i}"
                )
            with col2:
                if st.button("💾 Save", key=f"save_{i}", use_container_width=True):
                    st.session_state.history[i]["Label"] = new_label
                    st.session_state.corrections[item["News"]] = new_label
                    st.success(f"✅ Updated to {new_label}")
                    st.rerun()

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
                with st.spinner("🧠 Generating explanation..."):
                    try:
                        explanation = explain_fake_news(item["News"][:1000])  # Limit length for API
                        st.markdown("#### 🧠 Why This May Be Fake")
                        st.info(explanation)
                    except Exception as e:
                        st.error(f"Could not generate explanation: {str(e)}")

            st.markdown("</div>", unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("""
<div class='footer'>
Built with <strong>Streamlit</strong> • <strong>HuggingFace Transformers</strong> • <strong>Groq LLaMA</strong><br><br>
<strong>🔬 Analysis Pipeline:</strong><br>
1️⃣ Smart input detection (URL vs Text)<br>
2️⃣ Automatic article extraction from URLs<br>
3️⃣ Source credibility validation<br>
4️⃣ AI model classification<br>
5️⃣ Hallucination keyword detection<br>
6️⃣ LLM fact-checking ensemble<br>
7️⃣ Manual correction & learning<br>
8️⃣ AI-generated explanations for fake news<br>
</div>
""", unsafe_allow_html=True)