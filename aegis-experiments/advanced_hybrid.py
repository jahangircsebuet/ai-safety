from idea_2_token_level_blocking import SafetyFilterLogitsProcessor
from transformers import LogitsProcessorList
import torch

class AdvancedHybridMethod:
    def __init__(self):
        pass
    def hybrid_safe_generate(model, tokenizer, safety_ensemble, prompt,
                            num_beams=5, max_new_tokens=50, top_k=50,
                            strategy="any", fallback_attempts=5,
                            sampling_kwargs=None):
        """
        Hybrid decoding:
        (1) Try beam search with token-level blocking (SafeLogitsProcessor).
        (2) If all beams blocked or unsafe, fall back to rejection sampling.
        
        :param model: HuggingFace causal language model.
        :param tokenizer: Corresponding tokenizer.
        :param safety_ensemble: SafetyEnsemble instance.
        :param prompt: User input string.
        :param num_beams: Number of beams for beam search.
        :param max_new_tokens: Maximum generation length.
        :param top_k: Top-K filtering threshold in LogitsProcessor.
        :param strategy: Safety voting strategy ("any", "majority", "all").
        :param fallback_attempts: How many sampling tries in fallback.
        :param sampling_kwargs: Sampling params (temp, top_p, etc).
        :return: Safe output string or None.
        """
        

        # 1. Token-level blocking via LogitsProcessor during beam search
        safety_filter = SafetyFilterLogitsProcessor(safety_ensemble, tokenizer, strategy=strategy, top_k=top_k)
        logits_processors = LogitsProcessorList([safety_filter])
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)

        # Try beam decoding with token-level blocking
        try:
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                logits_processor=logits_processors,
                early_stopping=True
            )
            candidates = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            safe_candidates = [text for text in candidates if not safety_ensemble.is_unsafe(text, strategy=strategy)]
            if safe_candidates:
                return safe_candidates[0]  # return top-ranked safe beam
            print("❌ All beam candidates blocked or unsafe. Falling back to sampling...")
        except Exception as e:
            print(f"⚠️ Beam decoding failed: {e}")
            print("Falling back to sampling...")

        # 2. Fallback to rejection sampling
        if sampling_kwargs is None:
            sampling_kwargs = {'do_sample': True, 'temperature': 0.7, 'top_p': 0.9}
        for attempt in range(1, fallback_attempts + 1):
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                **sampling_kwargs
            )
            text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            if not safety_ensemble.is_unsafe(text, strategy=strategy):
                print(f"✅ Found safe output via sampling (attempt {attempt})")
                return text
        print("❌ All fallback sampling attempts failed.")
        return None
