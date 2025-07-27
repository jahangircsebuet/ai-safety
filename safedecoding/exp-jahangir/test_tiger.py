# https://www.kaggle.com/code/jahangir67/tigerllm-testing-and-benchmarking/edit

# TigerLLM 1B Comprehensive Testing & Benchmarking Notebook

# Author: AI Model Evaluation
# Model: md-nishat-008/TigerLLM-1B-it (Bengali Language Model)
# Purpose: Comprehensive capability assessment, benchmarking, and usage recommendations


# Cell 2: Model Loading and Configuration
# ======================================
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("🚀 Loading TigerLLM 1B Model...")
print("=" * 50)

# Model configuration
MODEL_NAME = "md-nishat-008/TigerLLM-1B-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,  trust_remote_code=True, token='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')

# Load Llama 3.2 1B tokenizer for TigerLLM 1B variant
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    use_fast=False,
    trust_remote_code=True,
    token='<hf_token>'
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=DEVICE,
    trust_remote_code=True,
    token='<hf_token>'
)


# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
# model_llama32 = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")


# tokenizer = AutoTokenizer.from_pretrained(
#     "meta-llama/Llama-3-1B",  # replace with correct model ID if it exists
#     use_fast=False,
#     trust_remote_code=True,
#     token='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'
# )
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     device_map="cuda",
#     trust_remote_code=True,
#     token='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'
# )

# Model statistics
param_count = sum(p.numel() for p in model.parameters())
print(f"📊 Model Parameters: {param_count:,} ({param_count/1e9:.2f}B)")
print(f"💾 Memory Usage: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"⚡ Device: {DEVICE}")
print(f"🔧 Model loaded successfully!")
