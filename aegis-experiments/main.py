from safety_ensemble import SafetyEnsemble
from idea_1_safe_decoding import SafeOutputGenerator
from load_base_models import LoadBaseModel
from idea_2_token_level_blocking import SafetyFilterLogitsProcessor
from hybrid_beam_search_token_level_blocking import HybridMethod
from advanced_hybrid import AdvancedHybridMethod
import torch

# pip install huggingface_hub


# IDEA 1 
# Example usage for Safe Decoding:
# In the above example, we attempt to generate a response to a potentially unsafe request. Beam search re-ranking will ideally find that all straightforward 
# completions are disallowed (the safety ensemble should flag direct instructions to make a bomb as unsafe). If all beams are unsafe, we fall back to 
# rejection sampling, which might produce a refusal or a safe response (due to the model's internal safety or randomness producing a milder answer). 
# The final safe_response should be a response that does not violate safety (for example, the model might refuse: "I'm sorry, I cannot assist with that 
# request.").

# This approach of Best-of-N decoding (beam or diverse sampling) combined with a safety filter is an effective inference-time alignment technique
# arxiv.org
# . It ensures we choose an answer that satisfies the user (if possible) while adhering to safety policies, without modifying the model's weights.


# waiting for the approval from Llama repo in hugging face 
# safety_classifiers, safety_tokenizer, model_llama2, tokenizer_llama2 = LoadBaseModel.load_model()

# now using the non gated model mistral - mistral is also gated repo - waiting for approval
# safety_classifiers, safety_tokenizer, model_mistral, tokenizer_mistral = LoadBaseModel.load_model()

# loading the tiny llama model 
safety_classifiers, safety_tokenizer, model, tokenizer = LoadBaseModel.load_model()

# Initialize the safety ensemble with the loaded classifiers
safety_ensemble = SafetyEnsemble(safety_classifiers, safety_tokenizer, default_strategy="any")

# waiting for the approval from the Llama gated model from hugging face repo 
# # Choose a base model for generation (either LLaMA-2 or Mistral)
# model = model_llama2  # using LLaMA-2-13B in this example
# tokenizer = tokenizer_llama2


# mistral also gated repo - waiting for approval
# now using the nongated model mistral 
# Choose a base model for generation (either LLaMA-2 or Mistral)
# model = model_mistral
# tokenizer = tokenizer_mistral

# there are other non gated models
# mistralai/Mistral-7B-v0.1 -> 7B  -> Mistral-7B
# mistralai/Mixtral-8x7B-v0.1 -> 12.9B active -> Mixtral-8x7B (MoE)
# teknium/OpenHermes-2.5-Mistral-7B -> 7B -> OpenHermes-2.5-Mistral (chat fine tuned mistral)
# TinyLlama/TinyLlama-1.1B-Chat-v1.0 -> 1.1B -> TinyLLaMA (very small and fast for testing, smallscale prototyping)
# microsoft/phi-2 -> 2.7B -> Phi-2 (lightweight, fast for prototyping, smallscale prototyping)

print("IDEA 1->using SafeOutputGenerator - picking the safest beam candidate")
safe_generator = SafeOutputGenerator(model, tokenizer, safety_ensemble)

prompt = "User: I want to insult someone.\nAssistant:"  # an example unsafe prompt
safe_response = safe_generator.generate_beam(prompt, num_beams=5, max_new_tokens=50)
if safe_response is None:
    print("All beam candidates were unsafe. Trying rejection sampling...")
    safe_response = safe_generator.generate_rejection(prompt, max_attempts=5, sampling_kwargs={'temperature': 0.7, 'top_p': 0.8})
print("Safe response:", safe_response)


print("IDEA 2 -> using LogiProcessor")
# IDEA 2 
# Example usage of SafetyFilterLogitsProcessor with Hugging Face generate

# Integrating the LogitsProcessor in generation: Using our SafetyFilterLogitsProcessor is straightforward with Hugging Face's generate API – 
# just pass it via the logits_processor argument. Here's an example:

safety_filter = SafetyFilterLogitsProcessor(safety_ensemble, tokenizer, strategy="any", top_k=50)
prompt = "User: I want to insult someone.\nAssistant:"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
output_ids = model.generate(input_ids, max_new_tokens=50, logits_processor=[safety_filter])
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Filtered output:", output_text)

# Explanation:
# In this example, as the model tries to generate a harmful insult, the safety filter will detect the partial output becoming unsafe and will block those 
# toxic words, forcing the model to choose a different continuation (likely resulting in a refusal or a toned-down response). The strategy="any" ensures 
# even one classifier's flag is enough to censor a token (strict mode), but you could use a majority vote to be slightly more lenient (to reduce false 
# positives where one classifier might be over-sensitive).

# if False:
print("IDEA 2 alternate - token level blocking")
# Alternte of calling model.generate - modifying the decoding loop 
partial_text = prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
for step in range(50):  # generate up to 50 tokens
    # Get raw logits for next token
    outputs = model(input_ids)
    next_token_logits = outputs.logits[:, -1, :]
    # Filter logits using safety ensemble                
    safe_logits = safety_filter.filter_logits_with_safety(next_token_logits[0], partial_text, safety_ensemble, tokenizer)
    # Choose the token with highest safe probability
    # safe_token_id = int(torch.tensor(safe_logits).argmax())
    safe_token_id = int(safe_logits.argmax())
    # If EOS token, break
    if safe_token_id == tokenizer.eos_token_id:
        break
    # Append token to sequence
    new_token = torch.tensor([[safe_token_id]]).to(input_ids.device)
    input_ids = torch.cat([input_ids, new_token], dim=1)
    partial_text += tokenizer.decode([safe_token_id], clean_up_tokenization_spaces=False)
# partial_text now contains the generated safe completion
print("Safely generated text:", partial_text)


# This manual loop does the same thing: it gets the model's predicted logits, filters out any that would lead to unsafe content, picks the top remaining token, and appends it. By repeating this, you build a response token by token, never allowing the output to cross into disallowed content. This approach is similar in spirit to the FUDGE algorithm (which guides generation using a future discriminator) and other controlled generation techniques
# arxiv.org
#  – here our discriminator is the safety ensemble that steers the output.


# Conclusion:
# We have implemented two complementary decode-time safety techniques:
# Safe decoding (output-level filtering): Generate multiple outputs and use an ensemble of LLM-based safety classifiers to choose a safe result, or retry until success. This leverages techniques like best-of-N generation and rejection sampling to enforce safety post-hoc on model outputs
# arxiv.org
# .
# Token-level blocking (online filtering): Integrate safety classifiers into the generation process to preemptively block unsafe tokens. This uses Hugging Face's logits processing API to guide the model away from toxic or policy-violating completions in real time
# huggingface.co.


# These methods keep the model's weights fixed and perform inference-time alignment, which is flexible and doesn't require retraining the base model. By using an ensemble (LlamaGuard Permissive/Defensive and NeMo-Defensive), we combine different safety "experts" to increase robustness, as suggested by NVIDIA's AEGIS research
# huggingface.co
# arxiv.org
# . The code is modular and can be integrated into a larger system; for instance, one could first attempt safe beam search, and if that fails, fall back to token-level filtering as a stricter measure.


# call hybrid generator 
prompt = "User: I want to insult someone.\nAssistant:"
hybrid_output = HybridMethod().hybrid_safe_generate_beam(model, tokenizer, safety_ensemble, prompt)
print("Hybrid output:", hybrid_output)

# Explanation of hybrid method:
# Uses HuggingFace’s native generate method with LogitsProcessorList including your safety filter.
# Every next-token probability is filtered using the token-level blocking logic, at every decoding step.
# Runs beam search to explore multiple diverse paths simultaneously.
# No post-hoc filtering is needed — all generated sequences are safe by construction.
# Returns the first safe high-likelihood candidate, or None if blocked.


# advanced hybrid method
prompt = "User: How can I attack someone?\nAssistant:"
safe_output = AdvancedHybridMethod().hybrid_safe_generate(
    model, tokenizer, safety_ensemble,
    prompt=prompt
)
print("Hybrid Safe Output:", safe_output)

# explanation of advanced hybrid method 
# | Phase             | Behavior                                                                 |
# | ----------------- | ------------------------------------------------------------------------ |
# | Beam Search       | Runs `num_beams` decoding paths, filters unsafe tokens during generation |
# | Safety Filter     | Token-level filtering blocks known unsafe completions early              |
# | Fallback Strategy | Uses stochastic sampling (temp + top\_p) if beam fails                   |
# | Final Output      | Returns highest-probability **safe** candidate                           |




# measure performance 
# how safe
# how fast - time complexity 

# post inference - will be slower

# inference/decode level 
# beam and token level are inference time 


# decode level - is the token level blocking 
# post inference - beam candidate


# why we are doing the hybrid approach 