"""
Query handler module for processing questions with different models.
"""
import time
from langchain_core.messages import SystemMessage, HumanMessage
from model_config import get_model, calculate_cost, MODEL_PRICING


def query_model(model_name, context, question):
    """
    Query a model with context and a question, tracking metrics.

    Args:
        model_name: Name of the model to query
        context: Full document context
        question: User question

    Returns:
        Dict with response, metrics, and timing information
    """
    # Initialize model
    model = get_model(model_name)

    # Prepare messages
    system_prompt = f"""Use the given context to answer the question.
If you don't know the answer, say you don't know. Keep the answer concise.

Context:
{context}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]

    # Track timing
    start_time = time.time()

    # Query model
    try:
        response = model.invoke(messages)
        elapsed_time = time.time() - start_time

        # Extract token usage
        if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
            token_usage = response.response_metadata['token_usage']
            input_tokens = token_usage.get('prompt_tokens', 0)
            output_tokens = token_usage.get('completion_tokens', 0)
        elif hasattr(response, 'usage_metadata'):
            # LangChain's unified format
            input_tokens = response.usage_metadata.get('input_tokens', 0)
            output_tokens = response.usage_metadata.get('output_tokens', 0)
        else:
            input_tokens = 0
            output_tokens = 0

        # Calculate cost
        cost = calculate_cost(model_name, input_tokens, output_tokens)

        return {
            "model": MODEL_PRICING[model_name]["name"],
            "response": response.content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": cost,
            "time": elapsed_time,
            "error": None
        }

    except Exception as e:
        elapsed_time = time.time() - start_time

        # Create more helpful error messages
        error_msg = str(e)
        if "402" in error_msg or "Insufficient Balance" in error_msg:
            error_msg = "Insufficient Balance - Please add credits to your DeepSeek account at https://platform.deepseek.com"
        elif "401" in error_msg or "Unauthorized" in error_msg:
            error_msg = "Invalid API key - Please check your API key in .env file"

        return {
            "model": MODEL_PRICING[model_name]["name"],
            "response": f"Error: {error_msg}",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost": 0,
            "time": elapsed_time,
            "error": error_msg
        }
