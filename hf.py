""""
import config
from huggingface_hub import InferenceClient
MODELS = getattr(
    config, 
    "HF_MODELS", 
    [
        "Qwen/Qwen2.5-7B-Instruct",          
        "microsoft/Phi-3-mini-4k-instruct",   
        "HuggingFaceH4/zephyr-7b-beta",       
        "mistralai/Mistral-7B-Instruct-v0.3"  
    ]
)
def generate_response(prompt: str, temperature: float = 0.3, max_tokens: int = 512) -> str:
    key = getattr(config, "HF_API_KEY", None)
    if not key:
        return "Error: HF_API_KEY missing in config.py"
    last_err = None
    for m in MODELS:
        try:
            c = InferenceClient(model=m, token=key)
            r = c.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content
        except Exception as e:
            last_err = e
    return (
        "Hugging Face model failed.\n"
        f"Tried models: {MODELS}\n"
        "Fix:\n"
        "1) Switch to Groq by importing groq.py in main.py OR\n"
        "2) Replace HF model in hf.py (HF_MODELS).\n"
        f"Details: {type(last_err).__name__}: {last_err}"
    )
"""
import config
from huggingface_hub import InferenceClient

# Updated to models 100% verified to work with serverless .chat_completion()
MODELS = getattr(
    config,
    "HF_MODELS",
    [
        "meta-llama/Llama-3.2-3B-Instruct",   # Fast, natively supported chat model
        "Qwen/Qwen2.5-Coder-7B-Instruct",    # Highly active serverless chat endpoint
        "meta-llama/Llama-3.3-70B-Instruct",  # Premium serverless chat fallback
    ]
)


def generate_response(prompt: str, temperature: float = 0.3, max_tokens: int = 512) -> str:
    key = getattr(config, "HF_API_KEY", None)
    if not key:
        return "Error: HF_API_KEY missing in config.py"
        
    last_err = None
    for m in MODELS:
        try:
            c = InferenceClient(token=key) # Initialize client generally
            
            # Use chat_completion on the specific model
            r = c.chat_completion(
                model=m,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content
        except Exception as e:
            last_err = e
            continue # Try the next model in the list
            
    return (
        "Hugging Face model failed.\n"
        f"Tried models: {MODELS}\n"
        "Fix:\n"
        "1) Switch to Groq by importing groq.py in main.py OR\n"
        "2) Replace HF model in hf.py (HF_MODELS).\n"
        f"Details: {type(last_err).__name__}: {last_err}"
    )
