import os
import pandas as pd
import re
from transformers import pipeline
from llmhelper import explain_fake_news

def get_llm_classifier(model_name="afsanehm/fake-news-detection-llm"):
    """
    Returns a fake news classification pipeline.
    """
    return pipeline("text-classification", model=model_name)

def classify_with_llm(texts, classifier=None):
    if classifier is None:
        classifier = get_llm_classifier()
    results = classifier(texts, truncation=True)
    return [r['label'] for r in results]

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', str(text))
    return text.lower().strip()

def process_posts(raw_file_path):
    df = pd.read_csv(raw_file_path)

    # Clean text columns
    for col in ['text', 'title', 'content']:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Classify using Hugging Face fake news LLM
    if 'text' in df.columns:
        print(f"Classifying {raw_file_path} with fake-news LLM...")
        classifier = get_llm_classifier()
        df['llm_label'] = classify_with_llm(df['text'].astype(str).tolist(), classifier)

        # For rows labeled as FAKE, generate explanations via Groq LLaMA
        print("Generating explanations for FAKE news using Groq LLaMA...")
        df['llm_explanation'] = None  # Initialize column
        fake_rows = df['llm_label'] == 'FAKE'
        for idx in df[fake_rows].index:
            explanation = explain_fake_news(df.at[idx, 'text'])
            df.at[idx, 'llm_explanation'] = explanation

    # Save the cleaned and labeled file with explanations
    base = os.path.basename(raw_file_path)
    out_path = os.path.join(os.path.dirname(raw_file_path), f"cleaned_{base}")
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    process_posts("data/fake.csv")
    process_posts("data/true.csv")
