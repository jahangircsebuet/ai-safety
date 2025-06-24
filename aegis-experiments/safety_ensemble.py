import torch

class SafetyEnsemble:
    """
    Ensemble of safety classifier LLMs (e.g., LlamaGuard Permissive/Defensive, NeMo Defensive).
    Provides methods to assess content safety using multiple models.
    """
    def __init__(self, classifiers, tokenizer, device=None, default_strategy="any"):
        """
        Initialize the safety ensemble.
        :param classifiers: List of HuggingFace models for safety classification.
        :param tokenizer: Tokenizer corresponding to the classifier models (assumed same for all if based on Llama-2).
        :param device: Torch device (e.g., 'cuda') on which to run the classifiers. If None, use device of first model.
        :param default_strategy: Voting strategy to decide safety ("any", "majority", or "all").
        """
        self.classifiers = classifiers
        self.tokenizer = tokenizer
        self.device = device if device is not None else next(classifiers[0].parameters()).device
        self.default_strategy = default_strategy

        # Ensure all classifiers are in evaluation mode and on the correct device
        for clf in self.classifiers:
            clf.eval()
            # If device is specified and model is not on it, move or parallelize as needed
            # (For PEFT models, device_map was used at loading, so they should already be on device.)
    
    def is_unsafe(self, text, strategy=None):
        """
        Determine if the given text is unsafe according to the ensemble.
        :param text: Input text (prompt and/or model output) to evaluate.
        :param strategy: Voting strategy override (if None, uses default_strategy).
        :return: True if text is flagged as unsafe, False if deemed safe.
        """
        strat = strategy if strategy is not None else self.default_strategy
        # Encode the text for the classifier model(s)
        # (We can prepend any required system prompt or formatting if the classifier expects it.
        #  Here we assume the classifier was fine-tuned to take the raw text as input for moderation.)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        results = []  # collect boolean flags from each classifier
        for clf in self.classifiers:
            # Generate a short output from the classifier model
            with torch.no_grad():
                output_ids = clf.generate(**inputs, max_new_tokens=16)  # enough tokens to output "safe" or "unsafe"
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            # Determine if this model flagged the content as unsafe.
            # Simple heuristic: check if the word "unsafe" appears (the models are trained to output "unsafe" and a category ID if unsafe:contentReference[oaicite:6]{index=6}, or "safe" if not).
            verdict = output_text.strip().lower()
            if "unsafe" in verdict:
                results.append(True)
            elif "safe" in verdict:
                results.append(False)
            else:
                # If the output is unclear (e.g., empty or not explicit), default to not unsafe (False) or handle as needed.
                results.append(False)
        # Voting logic:
        unsafe_votes = sum(1 for flag in results if flag)
        if strat == "any":
            # Any one classifier flagging unsafe makes the result unsafe
            return unsafe_votes >= 1
        elif strat == "majority":
            # More than half of the classifiers must flag unsafe
            return unsafe_votes > len(results) / 2.0
        elif strat == "all":
            # All classifiers must agree unsafe
            return unsafe_votes == len(results)
        else:
            raise ValueError(f"Unknown strategy: {strat}")


# Explanation: The SafetyEnsemble.is_unsafe method runs each classifier model on the input text. We use model.generate to have the classifier produce an output string (each is an LLM acting as a classifier). According to NVIDIA's model card, the LlamaGuard models will output "unsafe" followed by a category ID if the text violates the policy, or "safe" if it does not
# huggingface.co
# . We parse the generated text to check for the word "unsafe". Based on all classifiers' outputs, we decide if the text is unsafe using the chosen strategy. For example, with the default "any" strategy, if any classifier outputs an unsafe verdict, we label the text as unsafe.
# Note: In a real system, one might use a more direct method (such as a classifier head or logits) to get a probability or score for safety from each model instead of generating text, especially to reduce latency. But for simplicity and generality, we treat them as generative classifiers here. Also, consider caching the safety model outputs if classifying many similar candidates to avoid redundant computation.