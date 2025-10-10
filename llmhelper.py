from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables or .env file")

llm = ChatGroq(api_key=API_KEY, model_name="meta-llama/llama-4-scout-17b-16e-instruct")

def explain_fake_news(text: str) -> str:
    """
    Use Groq LLaMA to explain why a text is classified as fake news.
    """
    prompt = (
        f"The following news article has been classified as FAKE.\n"
        f"Explain in a few sentences why this might be fake news:\n\n{text}\n\nExplanation:"
    )
    response = llm.invoke(prompt)
    return response.content.strip()

def explain_real_news(text: str) -> str:
    """
    Use Groq LLaMA to explain why a text is credible and reliable news.
    """
    prompt = (
        f"The following news article has been classified as REAL.\n"
        f"Explain in a few sentences why this news appears credible and reliable:\n\n{text}\n\nExplanation:"
    )
    response = llm.invoke(prompt)
    return response.content.strip()

if __name__ == "__main__":
    sample_fake = "Aliens landed on Earth yesterday and gave free technology to everyone."
    sample_real = "NASA successfully launched a new Mars rover last week."

    print("Fake news explanation:")
    print(explain_fake_news(sample_fake))
    print("\nReal news explanation:")
    print(explain_real_news(sample_real))
