# Idea 2: Token-Level Blocking with Safety Classifiers

# The second technique enforces safety during generation by disallowing unsafe tokens at each step. We achieve this by integrating the safety ensemble as a Logits Processor in the generation loop. A Hugging Face LogitsProcessor can modify the model's next-token probabilities on the fly
# huggingface.co
# . Our custom processor will use the ensemble to evaluate the partial output after adding a candidate token; if the partial sequence would be unsafe, that token's logit is set to negative infinity (blocking it).

# We support the same voting strategies ("any", "majority", "all") to decide if a partial sequence is unsafe. For example, under "any" strategy, if any classifier flags the content of the partial sequence as unsafe, the next token is blocked.

# Below is the implementation of the SafetyFilterLogitsProcessor and a generic utility function for non-Transformers pipelines:

from transformers import LogitsProcessor
import torch

class SafetyFilterLogitsProcessor(LogitsProcessor):
    """
    A HuggingFace LogitsProcessor that filters out tokens leading to unsafe content.
    It uses a SafetyEnsemble to check *************if the current generated text plus a candidate token is unsafe***********************.
    If unsafe (according to the chosen voting strategy), that token is blocked (its probability set to 0).
    """
    def __init__(self, safety_ensemble, tokenizer, strategy="any", top_k=50):
        """
        :param safety_ensemble: SafetyEnsemble object for classifying content.
        :param tokenizer: Tokenizer for decoding tokens to text.
        :param strategy: Voting strategy ("any", "majority", "all") for the ensemble to flag unsafe.
        :param top_k: (Optional) Limit checking to the top_k tokens to save time.
        """
        self.safety = safety_ensemble
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.top_k = top_k
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Process each sequence in the batch
        batch_size = scores.shape[0]
        # We will modify the scores in place for any tokens deemed unsafe
        for i in range(batch_size):
            # Decode the current generated text for sequence i
            prefix_text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
            # Get indices of tokens sorted by score (high to low) for efficiency
            sorted_indices = torch.argsort(scores[i], descending=True)
            # (Optionally restrict to top_k candidates to check, to reduce classifier calls)
            if self.top_k is not None:
                sorted_indices = sorted_indices[:self.top_k]
            # Iterate through candidate tokens in order of likelihood
            for token_id in sorted_indices.tolist():
                # Form the candidate text by appending this token to the prefix
                candidate_text = prefix_text + self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                # Check safety of the candidate sequence
                if self.safety.is_unsafe(candidate_text, strategy=self.strategy):
                    # If unsafe, block this token by setting an extremely low score
                    scores[i, token_id] = -float('inf')
                else:
                    # If this token is safe, we stop checking further down this list for this sequence.
                    # (We don't block lower-probability tokens that haven't been checked; they will be checked if they become top during generation.)
                    # Breaking here ensures we only block obvious unsafe tokens and allow the highest-ranked safe token to be chosen.
                    break
        return scores

    def filter_logits_with_safety(self, logits, partial_text, safety_ensemble, tokenizer, strategy="any"):
        """
        A helper function to filter logits according to safety outside the HF generation loop.
        Given the current partial text and next-token logits, it zeroes out probabilities of unsafe continuations.
        :param logits: Numpy or torch 1D array of next-token scores (size = vocab_size).
        :param partial_text: The text generated so far (prefix).
        :param safety_ensemble: SafetyEnsemble for checking safety.
        :param tokenizer: Tokenizer to decode token ids.
        :param strategy: Voting strategy for the safety ensemble.
        :return: Modified logits (numpy array) with unsafe token probabilities set to 0.
        """
        # Convert logits to numpy for ease of manipulation (if it's a PyTorch tensor)
        # if isinstance(logits, torch.Tensor):
        #     logits = logits.cpu().detach().numpy()
        # Get indices sorted by score
        # sorted_indices = logits.argsort()[::-1]
        print(type(logits), logits.shape)
        # Ensure logits is a 1D PyTorch tensor on CPU
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().squeeze()
        else:
            raise ValueError("Expected logits to be a PyTorch tensor.")

        # Sort token indices by descending score
        sorted_indices = torch.argsort(logits, descending=True)
        for token_id in sorted_indices:
            token_str = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            candidate_text = partial_text + token_str
            print("token_str: ", token_str)
            print("candidate_text: ", candidate_text)
            
            if safety_ensemble.is_unsafe(candidate_text, strategy=strategy):
                logits[token_id] = -float('inf')  # block this token
            else:
                break  # stop at the first safe token candidate
        return logits


# Notes on implementation:
# In SafetyFilterLogitsProcessor.__call__, we limit checks to top_k highest probability tokens (by default 50) to reduce overhead. Checking the entire 
# vocabulary (which might be 32k tokens) for each step is costly. The assumption is that unsafe continuations are likely to involve certain high-probability 
# tokens (like certain words), and if none of the top_k tokens lead to unsafe content, we can allow the model to proceed with those. This is a trade-off 
# between safety and performance.

# We decode the current prefix text for each sequence in the batch and then each candidate token to form the new sequence. We then use SafetyEnsemble.is_unsafe to determine if that sequence would be unsafe. If it is, we set that token's score to -inf to block it. We break out as soon as we find a token that is safe, under the assumption that tokens ranked lower than a safe token won't be chosen if a safe token with higher score exists. This ensures we don't unnecessarily block more than needed.
# filter_logits_with_safety is a utility doing essentially the same logic outside of the HF generate context. This could be used if you're implementing a custom generation loop or using a different library. It takes raw logits and the current generated text, and filters out unsafe tokens.