from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="meta-llama/llama-4-scout-17b-16e-instruct")

def explain_fake_news(text):
    """
    Use Groq LLaMA to explain why a text is fake news.
    """
    prompt = (
        f"The following news article has been classified as FAKE.\n"
        f"Explain in a few sentences why this might be fake news:\n\n{text}\n\nExplanation:"
    )
    response = llm.invoke(prompt)
    return response.content.strip()

if __name__ == "__main__":
    sample = "AI is taking over the world next year."
    print("Sample explanation:")
    print(explain_fake_news(sample))
