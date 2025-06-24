# Idea 1: Safe Decoding via Beam Reranking and Rejection Sampling
# Safe decoding generates multiple candidate outputs and then selects a result that meets safety criteria. 
# We implement two strategies for this: 
# (1) Beam Search Re-ranking: Generate N candidates using beam search, then pick the highest-probability candidate that is safe. Beam search explores multiple high-probability continuations in parallel. By re-ranking with a safety model (our ensemble), we can avoid unsafe completions even if they had highest likelihood.
# (2) Rejection Sampling: Generate outputs (via sampling/stochastic decoding) repeatedly until a safe output is found, up to a maximum number of tries. This leverages randomness to potentially find a safe completion even if the most likely completion is unsafe.

import time

class SafeOutputGenerator:
    """
    Generator that produces safe outputs from a base language model using decode-time strategies.
    Provides beam search re-ranking and rejection sampling to ensure outputs pass safety checks.
    """
    def __init__(self, model, tokenizer, safety_ensemble):
        """
        :param model: HuggingFace CausalLM model for text generation (e.g., LLaMA-2-13B or Mistral-7B).
        :param tokenizer: Tokenizer corresponding to the model.
        :param safety_ensemble: SafetyEnsemble instance for evaluating output safety.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.safety = safety_ensemble
        # Ensure model is in eval mode
        self.model.eval()
    
    def generate_beam(self, prompt, num_beams=5, max_new_tokens=100, **gen_kwargs):
        """
        Generate text using beam search and return the safest high-probability output.
        :param prompt: Input prompt string.
        :param num_beams: Number of beams for beam search.
        :param max_new_tokens: Maximum tokens to generate for each beam.
        :param gen_kwargs: Additional keyword args for model.generate (e.g., stopping criteria).
        :return: A safe output string if found, otherwise None (if all beams are unsafe).
        """
        # Tokenize prompt and generate multiple beams
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(next(self.model.parameters()).device)
        # Use beam search to generate multiple candidate outputs
        output = self.model.generate(input_ids, num_beams=num_beams, num_return_sequences=num_beams, 
                                     max_new_tokens=max_new_tokens, **gen_kwargs)
        # Decode all candidate outputs
        candidates = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in output]
        # Evaluate safety of each candidate
        safe_candidates = []
        for text in candidates:
            if not self.safety.is_unsafe(text):
                safe_candidates.append(text)
        if safe_candidates:
            # If there are any safe candidates, choose the one that corresponds to the highest beam score.
            # Since generate with beam search returns sequences sorted by descending likelihood, 
            # the first in `candidates` list is the most likely. We will return the first safe candidate 
            # in that list, which ensures we return the highest-probability safe output.
            for text in candidates:  # iterate in original order (which is sorted by score)
                if text in safe_candidates:
                    return text
        # If no safe candidate found, return None (could also choose to return the "least unsafe" if desired)
        return None
    
    def generate_rejection(self, prompt, max_attempts=5, sampling_kwargs=None, **gen_kwargs):
        """
        Generate text using rejection sampling until a safe output is found or attempts exhausted.
        :param prompt: Input prompt string.
        :param max_attempts: Maximum number of generation attempts to try.
        :param sampling_kwargs: Dict of generation args for sampling (e.g., temperature, top_p).
        :param gen_kwargs: Additional kwargs for model.generate.
        :return: A safe output string if found, otherwise None.
        """
        if sampling_kwargs is None:
            sampling_kwargs = {'do_sample': True, 'temperature': 1.0, 'top_p': 0.9}
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(next(self.model.parameters()).device)
        for attempt in range(1, max_attempts+1):
            output = self.model.generate(input_ids, max_new_tokens=gen_kwargs.get('max_new_tokens', 100), 
                                         **sampling_kwargs, **gen_kwargs)
            text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            if not self.safety.is_unsafe(text):
                return text  # return on first safe generation
        # If we reach here, no safe output was found within the attempts
        return None
    
    def generate_batch(self, prompts, method="beam", **kwargs):
        """
        Generate outputs for a batch of prompts with safety filtering.
        :param prompts: List of prompt strings.
        :param method: "beam" or "rejection" to choose the decoding strategy.
        :param kwargs: Additional args to pass to generate_beam or generate_rejection.
        :return: List of outputs (safe texts or None if not found for that prompt).
        """
        outputs = []
        start_time = time.time()
        for prompt in prompts:
            if method == "beam":
                result = self.generate_beam(prompt, **kwargs)
            elif method == "rejection":
                result = self.generate_rejection(prompt, **kwargs)
            else:
                raise ValueError("method must be 'beam' or 'rejection'")
            outputs.append(result)
        elapsed = time.time() - start_time
        print(f"Processed {len(prompts)} prompts with {method} decoding in {elapsed:.2f} seconds.")
        return outputs


# Documentation
# How it works: The SafeOutputGenerator provides two main methods:
# (1) generate_beam: uses model.generate with num_beams beams and returns the first safe output found among the beams. 
# The beams are generated in descending order of log-probability (most likely first), so by iterating in order and picking the first safe, 
# we get the safest (highest probability) valid completion. If none of the beams are safe, it returns None (indicating all candidates were unsafe).
# (2) generate_rejection: uses random sampling (do_sample=True) to generate an output, checks it for safety, and if unsafe, tries again up to max_attempts. 
# We can adjust temperature, top_p, etc., via sampling_kwargs to control diversity. If a safe output is found, it's returned immediately; 
# otherwise None after exhausting attempts.

# The generate_batch method is a convenience to handle multiple prompts. In this simple implementation, it loops sequentially, but you could enhance it 
# to do true batch generation (padding inputs and using vectorized generation) for efficiency. It also prints the total time taken for processing 
# (latency tracking).
