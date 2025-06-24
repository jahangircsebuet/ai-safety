from transformers import LogitsProcessorList
from idea_2_token_level_blocking import SafetyFilterLogitsProcessor
import torch

class HybridMethod:
    def __init__(self):
        pass
    
    def hybrid_safe_generate_beam(model, tokenizer, safety_ensemble, prompt, num_beams=5, max_new_tokens=50, top_k=50, strategy="any"):
        """
        Perform beam search with token-level safety blocking (hybrid strategy).
        :param model: The base LLM.
        :param tokenizer: Tokenizer for the model.
        :param safety_ensemble: Safety ensemble instance.
        :param prompt: Input prompt string.
        :param num_beams: Number of beams for beam search.
        :param max_new_tokens: Maximum length of the generation.
        :param top_k: Token-level filter applies only to top_k tokens.
        :param strategy: Safety ensemble voting strategy.
        :return: The safest high-likelihood generation.
        """

        

        # Prepare safety filter
        safety_filter = SafetyFilterLogitsProcessor(safety_ensemble, tokenizer, strategy=strategy, top_k=top_k)
        logits_processors = LogitsProcessorList([safety_filter])

        # Encode the input prompt
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)

        # Perform beam search with token-level filtering
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            logits_processor=logits_processors,
            early_stopping=True,
        )

        # Decode all beams
        candidates = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        # Optionally, evaluate outputs using safety ensemble again (paranoid mode)
        safe_candidates = [text for text in candidates if not safety_ensemble.is_unsafe(text, strategy=strategy)]

        if safe_candidates:
            return safe_candidates[0]  # return highest-ranked safe beam
        else:
            return None
