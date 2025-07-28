# Cell 3: Core Testing Infrastructure
# ==================================
import json
import time
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List

class TigerLLMTester:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.results = {}
        self.test_start_time = datetime.now()
        
    def generate_text(self, prompt: str, max_new_tokens: int = 512, 
                     temperature: float = 0.95, top_p: float = 0.9, 
                     do_sample: bool = True, return_metrics: bool = False) -> Dict:
        """Enhanced text generation with metrics"""
        start_time = time.time()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to(self.device)
            
            # Clear any existing cache to avoid conflicts
            if hasattr(self.model, 'clear_cache'):
                self.model.clear_cache()
            
                        # Try with cache first, fallback to no cache if error occurs
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.eos_token_id,
                        attention_mask=inputs.attention_mask,
                        use_cache=True,
                        return_dict_in_generate=False
                    )
            except ValueError as cache_error:
                if "Max cache length is not consistent" in str(cache_error):
                    print(f"⚠️ Cache error detected, retrying without cache...")
                    # Retry without cache
                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs.input_ids,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=do_sample,
                            pad_token_id=self.tokenizer.eos_token_id,
                            attention_mask=inputs.attention_mask,
                            use_cache=False
                        )
                else:
                    raise cache_error
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_text = generated_text[len(prompt):].strip()
            
            # Calculate metrics
            generation_time = time.time() - start_time
            input_tokens = len(inputs.input_ids[0])
            output_tokens = len(outputs[0]) - input_tokens
            tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
            
            result = {
                'prompt': prompt,
                'full_text': generated_text,
                'generated_text': new_text,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'generation_time': generation_time,
                'tokens_per_second': tokens_per_second,
                'temperature': temperature,
                'top_p': top_p
            }
            
            return result
            
        except Exception as e:
            print(f"⚠️ Generation error: {e}")
            generation_time = time.time() - start_time
            
            # Return error result
            return {
                'prompt': prompt,
                'full_text': '',
                'generated_text': '',
                'input_tokens': 0,
                'output_tokens': 0,
                'generation_time': generation_time,
                'tokens_per_second': 0,
                'temperature': temperature,
                'top_p': top_p,
                'error': str(e)
            }
    
    def batch_test(self, prompts: List[str], test_name: str, **kwargs) -> List[Dict]:
        """Run batch testing with progress tracking"""
        results = []
        print(f"🧪 Running {test_name} ({len(prompts)} tests)...")
        
        for i, prompt in enumerate(prompts):
            print(f"  Progress: {i+1}/{len(prompts)}", end="\r")
            result = self.generate_text(prompt, **kwargs)
            results.append(result)
        
        self.results[test_name] = results
        print(f"✅ {test_name} completed!")
        return results
    
    def evaluate_quality(self, results: List[Dict], criteria: List[str]) -> Dict:
        """Evaluate generation quality based on criteria"""
        scores = {}
        
        for criterion in criteria:
            if criterion == 'fluency':
                # Basic fluency check (sentence structure, grammar indicators)
                scores[criterion] = self._assess_fluency(results)
            elif criterion == 'coherence':
                # Coherence assessment
                scores[criterion] = self._assess_coherence(results)
            elif criterion == 'relevance':
                # Relevance to prompt
                scores[criterion] = self._assess_relevance(results)
            elif criterion == 'length_consistency':
                # Length consistency
                scores[criterion] = self._assess_length_consistency(results)
        
        return scores
    
    def _assess_fluency(self, results: List[Dict]) -> float:
        """Assess text fluency (basic Bengali text quality indicators)"""
        total_score = 0
        for result in results:
            text = result['generated_text']
            score = 0
            
            # Check for basic structure
            if len(text) > 10: score += 0.3
            if '।' in text or '?' in text or '!' in text: score += 0.2
            if not text.startswith(' ') and len(text) > 0: score += 0.2
            if len(text.split()) > 3: score += 0.3
            
            total_score += min(score, 1.0)
        
        return total_score / len(results) if results else 0
    
    def _assess_coherence(self, results: List[Dict]) -> float:
        """Assess text coherence"""
        total_score = 0
        for result in results:
            text = result['generated_text']
            score = 0
            
            # Basic coherence indicators
            words = text.split()
            if len(words) > 5: score += 0.4
            if len(set(words)) / len(words) > 0.7: score += 0.3  # Word diversity
            if not any(word.count(char) > 5 for word in words for char in word): score += 0.3
            
            total_score += min(score, 1.0)
        
        return total_score / len(results) if results else 0
    
    def _assess_relevance(self, results: List[Dict]) -> float:
        """Assess relevance to prompt"""
        # Simplified relevance check
        return 0.75  # Placeholder - would need more sophisticated analysis
    
    def _assess_length_consistency(self, results: List[Dict]) -> float:
        """Assess length consistency"""
        lengths = [len(result['generated_text'].split()) for result in results]
        if not lengths:
            return 0
        
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        cv = std_length / mean_length if mean_length > 0 else 1
        
        return max(0, 1 - cv)  # Lower coefficient of variation = better consistency
    
    def save_results_to_json(self, filename: str = "tigerllm_results.json"):
        """Save all test results to JSON file with test_name, prompt, completion structure"""
        json_results = []
        
        for test_name, test_results in self.results.items():
            for result in test_results:
                json_item = {
                    'test_name': test_name,
                    'prompt': result['prompt'],
                    'completion': result['generated_text']
                }
                json_results.append(json_item)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Results saved to {filename} ({len(json_results)} items)")
        return json_results
    
    def save_evaluation_results_to_json(self, filename: str = "tigerllm_evaluation_results.json"):
        """Save all test results with evaluation scores to JSON file"""
        json_results = []
        
        for test_name, test_results in self.results.items():
            # Calculate evaluation scores for this test
            evaluation_scores = self.evaluate_quality(test_results, ['fluency', 'coherence', 'relevance', 'length_consistency'])
            
            for result in test_results:
                json_item = {
                    'test_name': test_name,
                    'prompt': result['prompt'],
                    'completion': result['generated_text'],
                    'fluency_score': evaluation_scores.get('fluency', 0.0),
                    'coherence_score': evaluation_scores.get('coherence', 0.0),
                    'relevance_score': evaluation_scores.get('relevance', 0.0),
                    'length_consistency_score': evaluation_scores.get('length_consistency', 0.0)
                }
                json_results.append(json_item)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Evaluation results saved to {filename} ({len(json_results)} items)")
        return json_results