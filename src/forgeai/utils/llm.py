from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from src.forgeai.config.settings import settings
import os
from typing import Optional


def get_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    **kwargs
) -> BaseChatModel:
    """
    Safe LLM Factory for Groq + Gemini
    """
    prov = (provider or settings.llm_provider).lower()
    model = model_name or settings.llm_model
    temp = temperature if temperature is not None else settings.temperature

    if prov == "groq":
        # Some versions of langchain-groq use 'model_name' instead of 'model'
        try:
            return ChatGroq(
                model=model,                    # Try 'model' first
                temperature=temp,
                max_tokens=settings.max_tokens,
                groq_api_key=settings.groq_api_key or os.getenv("GROQ_API_KEY"),
                **kwargs
            )
        except TypeError:
            # Fallback if it complains about multiple values for 'model'
            return ChatGroq(
                model_name=model,               # Use model_name instead
                temperature=temp,
                max_tokens=settings.max_tokens,
                groq_api_key=settings.groq_api_key or os.getenv("GROQ_API_KEY"),
                **kwargs
            )
    
    elif prov in ["gemini", "google", "google_genai"]:
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temp,
            max_tokens=settings.max_tokens,
            google_api_key=settings.google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            **kwargs
        )
    
    else:
        # Safe fallback
        return ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=temp,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )


# Convenience functions
def get_fast_llm():
    """Fast model for routing and critic"""
    return get_llm(provider="groq", model_name="llama-3.3-70b-versatile", temperature=0.1)


def get_strong_llm():
    """Strong reasoning - Use Gemini"""
    return get_llm(provider="gemini", model_name="gemini-2.5-pro", temperature=0.3)


def get_creative_llm():
    """For report writing"""
    return get_llm(provider="gemini", model_name="gemini-2.5-pro", temperature=0.7)