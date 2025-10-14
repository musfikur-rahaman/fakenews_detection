import streamlit as st
from model import load_classifier, hybrid_classify
from llmhelper import explain_fake_news

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(page_title="Adaptive Fake News Detector", layout="wide")

# ---------- SESSION ----------
if "user_input" not in st.session_state: st.session_state.user_input = ""
if "history" not in st.session_state: st.session_state.history = []
if "feedback_keywords" not in st.session_state: st.session_state.feedback_keywords = []
if "page" not in st.session_state: st.session_state.page = "Home"
if "collapsed" not in st.session_state: st.session_state.collapsed = []

# ---------- NAVIGATION ----------
st.markdown("<div class='nav-container'>", unsafe_allow_html=True)
if st.button("🏠 Home", key="nav_home"): st.session_state.page="Home"
if st.button("📋 Classify", key="nav_classify"): st.session_state.page="Classify"
if st.button("📝 History", key="nav_history"): st.session_state.page="History"
st.markdown("</div>", unsafe_allow_html=True)

# ---------- HOME PAGE ----------
if st.session_state.page=="Home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Welcome to Adaptive Fake News Detector")
    st.write("""
    This dashboard uses AI + hallucination detection to classify news as FAKE or REAL.
    
    **Features:**  
    - Hybrid AI + keyword hallucination detection  
    - Explanations for FAKE news using LLaMA  
    - Adaptive session learning: system improves with your feedback  
    - Reset input and review session history  
    - Collapsible cards for each prediction
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- CLASSIFY PAGE ----------
elif st.session_state.page=="Classify":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Classify News")
    user_input = st.text_area("Enter news text:", height=180, value=st.session_state.user_input)
    
    col1, col2 = st.columns([1,2])
    with col1:
        if st.button("Reset Input"):
            st.session_state.user_input = ""
            st.rerun()
    with col2:
        if st.button("Run Classification"):
            if user_input.strip():
                st.session_state.user_input = user_input
                classifier = load_classifier()
                extra_keywords = '|'.join(st.session_state.feedback_keywords) if st.session_state.feedback_keywords else None
                label, score, halluc_flag = hybrid_classify(user_input, classifier, extra_keywords)

                # Add to history
                st.session_state.history.insert(0, {  # newest first
                    "News": user_input,
                    "Label": label,
                    "Confidence": f"{score*100:.2f}%",
                    "Hallucination": halluc_flag
                })
                st.session_state.collapsed.insert(0, True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- HISTORY PAGE ----------
elif st.session_state.page=="History":
    st.header("Session History")
    if st.session_state.history:
        for i, item in enumerate(st.session_state.history):
            collapsed = st.session_state.collapsed[i]
            color_class = "fake" if item["Label"]=="FAKE" else "real"
            with st.expander(f"{item['Label']} | Confidence: {item['Confidence']}", expanded=collapsed):
                st.markdown(f"<div class='card {color_class} card-content'>", unsafe_allow_html=True)
                st.write(f"**News:** {item['News']}")
                st.write(f"**Hallucination flag:** {item['Hallucination']}")
                
                if item["Label"]=="FAKE":
                    explanation = explain_fake_news(item["News"])
                    st.markdown("#### 🧠 Why This May Be Fake")
                    st.write(explanation)
                
                # Feedback inside card
                st.markdown("#### ✅ Correct Prediction?")
                feedback = st.radio("Correct label?", ["Yes", "No"], key=f"feedback_{i}")
                if feedback=="No":
                    correct_label = st.selectbox("Correct label should be:", ["FAKE","REAL"], key=f"correct_{i}")
                    if st.button("Update Prediction", key=f"update_{i}"):
                        st.session_state.history[i]["Label"] = correct_label
                        if correct_label=="FAKE":
                            st.session_state.feedback_keywords.extend(item["News"].split())
                        st.success("✅ Prediction updated and session learning applied!")
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No classifications yet. Go to Classify tab to add news predictions.")

# ---------- FOOTER ----------
st.markdown("""
<div class='footer'>
<hr>
Built with <strong>Streamlit</strong>, <strong>HuggingFace Transformers</strong>, and <strong>Groq LLaMA</strong>.<br>
<strong>How it works:</strong> 
1) AI model predicts FAKE/REAL news based on training data.<br>
2) Hallucination keyword checker flags sensational, conspiratorial, or pseudoscientific content.<br>
3) Users provide feedback to correct predictions; the system adapts and improves within the session.<br>
4) FAKE news gets a LLaMA-generated explanation.<br>
5) Modern collapsible card layout for easy navigation and review of each prediction.
</div>
""", unsafe_allow_html=True)
