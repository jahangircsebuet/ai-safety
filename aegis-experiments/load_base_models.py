from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # adjust CUDA version if needed
# pip install transformers
# pip install peft

class LoadBaseModel():
    def __init__(self):
        pass

    def load_model():

        # Load base text generation models (LLaMA-2-13B and Mistral-7B)
        llama2_model_id = "meta-llama/Llama-2-13b-hf"  # or "meta-llama/Llama-2-13b-chat-hf" for chat-tuned
        mistral_model_id = "mistralai/Mistral-7B-v0.1"  # base Mistral 7B model
        tinyllama_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        # there are other non gated models
        # mistralai/Mistral-7B-v0.1 -> 7B  -> Mistral-7B
        # mistralai/Mixtral-8x7B-v0.1 -> 12.9B active -> Mixtral-8x7B (MoE)
        # teknium/OpenHermes-2.5-Mistral-7B -> 7B -> OpenHermes-2.5-Mistral (chat fine tuned mistral)
        # TinyLlama/TinyLlama-1.1B-Chat-v1.0 -> 1.1B -> TinyLLaMA (very small and fast for testing, smallscale prototyping)
        # microsoft/phi-2 -> 2.7B -> Phi-2 (lightweight, fast for prototyping, smallscale prototyping)

        # Tokenizers
        tokenizer_llama2 = AutoTokenizer.from_pretrained(llama2_model_id)
        tokenizer_mistral = AutoTokenizer.from_pretrained(mistral_model_id)
        tokenizer_tinyllama = AutoTokenizer.from_pretrained(tinyllama_model_id)

        # Load models with half precision and auto device mapping for GPU (for memory efficiency, use 8-bit if needed)
        model_llama2 = AutoModelForCausalLM.from_pretrained(llama2_model_id, device_map="auto", torch_dtype=torch.float16)
        model_mistral = AutoModelForCausalLM.from_pretrained(mistral_model_id, device_map="auto", torch_dtype=torch.float16)
        model_tinyllama = AutoModelForCausalLM.from_pretrained(tinyllama_model_id, device_map="auto", torch_dtype=torch.float16)

        # Load base model for safety classifiers (LLaMA-2-7B)
        safety_base_id = "meta-llama/Llama-2-7b-hf"
        safety_tokenizer = AutoTokenizer.from_pretrained(safety_base_id)
        safety_base = AutoModelForCausalLM.from_pretrained(safety_base_id, device_map="auto", torch_dtype=torch.float16)

        # Load Aegis safety classifier LoRA adapters and apply to separate copies of the base model.
        # LlamaGuard Permissive 1.0 (less strict) and LlamaGuard Defensive 1.0 (more strict) - these models are trained using the Aegis Safety Dataset.
        safety_model_permissive = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(safety_base_id, device_map="auto", torch_dtype=torch.float16),
            "nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0"
        )
        safety_model_defensive = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(safety_base_id, device_map="auto", torch_dtype=torch.float16),
            "nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0"
        )

        ##TODO
        # (Optional) Load NeMo 43B Defensive safety model if available.
        # This model is a 43B-parameter LLM adapted via LoRA:contentReference[oaicite:4]{index=4} for safety classification.
        # It may require using an 8-bit or 4-bit load to fit in memory (and possibly multi-GPU).
        # Uncomment and adjust the path if you have the base and adapter:
        # nemo_base_id = "nvidia/nemo-43b-base"  # hypothetical base model name
        # safety_nemo_base = AutoModelForCausalLM.from_pretrained(nemo_base_id, device_map="auto", torch_dtype=torch.float16)
        # safety_model_nemo_def = PeftModel.from_pretrained(safety_nemo_base, "path/to/NeMo-43B-Defensive-LoRA")

        # Prepare the ensemble of safety classifiers
        safety_classifiers = [safety_model_permissive, safety_model_defensive]  # add safety_model_nemo_def if loaded
        # waiting for gated model Llama approval from hagging face 
        # return safety_classifiers, safety_tokenizer, model_llama2, tokenizer_llama2
        # now try using the non gated model mistral 
        
        # mistral is also gated repo 
        # return safety_classifiers, safety_tokenizer, model_mistral, tokenizer_mistral
    
        return safety_classifiers, safety_tokenizer, model_tinyllama, tokenizer_tinyllama
