"""
Model configuration module for different LLM providers.
"""
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import os
import requests


# Pricing per 1M tokens (input, output)
MODEL_PRICING = {
    "gpt-5": {
        "input": 2.50,
        "output": 10.00,
        "name": "GPT-5"
    },
    "claude-sonnet-4-5-20250929": {
        "input": 3.00,
        "output": 15.00,
        "name": "Claude Sonnet 4.5"
    },
    "deepseek-chat": {
        "input": 0.28,
        "output": 0.42,
        "name": "DeepSeek v3.2-Exp"
    },
    "deepseek-chat-v3.1": {
        "input": 0.55,
        "output": 2.19,
        "name": "DeepSeek v3.1-Terminus"
    },
}


def get_model(model_name):
    """
    Initialize and return a model instance.

    Args:
        model_name: Name of the model to initialize

    Returns:
        Initialized model instance
    """
    if model_name == "gpt-5":
        return ChatOpenAI(
            model="gpt-5",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    elif model_name == "claude-sonnet-4-5-20250929":
        return ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            temperature=0,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    elif model_name == "deepseek-chat":
        return ChatOpenAI(
            model="deepseek-chat",
            temperature=0,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
    elif model_name == "deepseek-chat-v3.1":
        return ChatOpenAI(
            model="deepseek-chat",
            temperature=0,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v3.1_terminus_expires_on_20251015"
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def calculate_cost(model_name, input_tokens, output_tokens):
    """
    Calculate the cost of an API call.

    Args:
        model_name: Name of the model used
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Cost in dollars
    """
    pricing = MODEL_PRICING[model_name]
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def check_deepseek_balance():
    """
    Check DeepSeek account balance.

    Returns:
        Dict with balance information or None if check fails
    """
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return None

        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(
            "https://api.deepseek.com/user/balance",
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None
