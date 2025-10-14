from transformers import pipeline
import re

MODEL_NAME = "afsanehm/fake-news-detection-llm"

# Load model once
def load_classifier():
    return pipeline("text-classification", model=MODEL_NAME)

# Hallucination keyword checker
def flag_hallucination_keywords(text, extra_keywords=None):
    patterns = [
        # Medical miracles / instant cures
        r'\bovernight\b|\binstantly\b|\bcures?\b|\bmiracle\b|\bdiabetes\b|\bcancer\b|\bAIDS\b',
        # Aliens, paranormal, supernatural
        r'\balien\b|\bpyramid\b|\bghost\b',
        # Money & scams
        r'\$\d{2,}|\bget rich\b|\bno work\b|\bpassive income\b|\bwin big\b|\bjackpot\b|\bmake money fast\b',
        # Celebrity / drama / scandal
        r'\bcelebrity\b|\bsecretly married\b|\bscandal\b|\bleaked\b',
        # Government & cover-ups
        r'\bcover(ing)? up\b|\bgovernment hiding\b|\bNASA\b|\bCIA\b|\bFBI\b|\bclassified\b',
        # Conspiracy & pseudoscience
        r'\b5G\b|\bvirus\b|\bvaccine\b|\bmicrochip\b|\bchemtrail\b|\bflat earth\b',
        r'\bscientists warn\b|\bexperts say\b|\bhidden truth\b|\bthey don’t want you to know\b|\bspread(s)? deadly\b',
    ]
    if extra_keywords:
        patterns.append(extra_keywords)
    return bool(re.search('|'.join(patterns), text, flags=re.IGNORECASE))

# Hybrid classification
def hybrid_classify(text, classifier, extra_keywords=None):
    result = classifier(text, truncation=True)[0]
    label = result["label"]
    score = result["score"]
    halluc_flag = flag_hallucination_keywords(text, extra_keywords)
    if halluc_flag:
        label = "FAKE"
        score = max(score, 0.97)
    return label, score, halluc_flag
