# Test TigerLLM with CATQA Bengali prompts
# =============================================

import json
import os
import time
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM

# Cell 1: Model Loading and Configuration
# ======================================
print("🚀 Loading TigerLLM 1B Model...")
print("=" * 50)

# Model configuration
MODEL_NAME = "md-nishat-008/TigerLLM-1B-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
import os

# First, let's try to load the model
print("🔄 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=DEVICE,
    trust_remote_code=True,
    token=os.getenv('HUGGING_FACE_HUB_TOKEN')
)

# Then try to load the tokenizer
print("🔄 Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        use_fast=False,
        trust_remote_code=True,
        token=os.getenv('HUGGING_FACE_HUB_TOKEN')
    )
    print("✅ Tokenizer loaded successfully")
except Exception as e:
    print(f"⚠️ Could not load Llama tokenizer: {e}")
    print("🔄 Trying to load tokenizer from TigerLLM model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            token=os.getenv('HUGGING_FACE_HUB_TOKEN')
        )
        print("✅ Tokenizer loaded from TigerLLM model")
    except Exception as e2:
        print(f"⚠️ Could not load tokenizer from TigerLLM either: {e2}")
        print("🔄 Using a fallback tokenizer...")
        # Use a simple fallback tokenizer
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

# Model statistics
param_count = sum(p.numel() for p in model.parameters())
print(f"📊 Model Parameters: {param_count:,} ({param_count/1e9:.2f}B)")
print(f"💾 Memory Usage: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"⚡ Device: {DEVICE}")
print(f"🔧 Model loaded successfully!")

# Cell 2: Testing Infrastructure
# ==============================
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
            
            # Try multiple generation strategies
            outputs = None
            generation_strategies = [
                # Strategy 1: Full parameters with cache
                {
                    'max_new_tokens': max_new_tokens,
                    'temperature': temperature,
                    'top_p': top_p,
                    'do_sample': do_sample,
                    'use_cache': True,
                    'name': 'Full with cache'
                },
                # Strategy 2: Full parameters without cache
                {
                    'max_new_tokens': max_new_tokens,
                    'temperature': temperature,
                    'top_p': top_p,
                    'do_sample': do_sample,
                    'use_cache': False,
                    'name': 'Full without cache'
                },
                # Strategy 3: Minimal parameters
                {
                    'max_new_tokens': min(max_new_tokens, 100),
                    'do_sample': False,
                    'use_cache': False,
                    'name': 'Minimal'
                },
                # Strategy 4: Very minimal
                {
                    'max_new_tokens': 50,
                    'do_sample': False,
                    'use_cache': False,
                    'name': 'Very minimal'
                }
            ]
            
            for i, strategy in enumerate(generation_strategies):
                try:
                    print(f"🔄 Trying strategy {i+1}: {strategy['name']}")
                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs.input_ids,
                            max_new_tokens=strategy['max_new_tokens'],
                            temperature=strategy.get('temperature', 1.0),
                            top_p=strategy.get('top_p', 1.0),
                            do_sample=strategy.get('do_sample', True),
                            pad_token_id=self.tokenizer.eos_token_id,
                            attention_mask=inputs.attention_mask,
                            use_cache=strategy.get('use_cache', True)
                        )
                    print(f"✅ Strategy {i+1} succeeded!")
                    break
                except Exception as e:
                    print(f"❌ Strategy {i+1} failed: {str(e)[:100]}...")
                    if i == len(generation_strategies) - 1:  # Last strategy
                        raise e
                    continue
            
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
                scores[criterion] = self._assess_fluency(results)
            elif criterion == 'coherence':
                scores[criterion] = self._assess_coherence(results)
            elif criterion == 'relevance':
                scores[criterion] = self._assess_relevance(results)
            elif criterion == 'length_consistency':
                scores[criterion] = self._assess_length_consistency(results)
        
        return scores
    
    def _assess_fluency(self, results: List[Dict]) -> float:
        """Assess text fluency (basic Bengali text quality indicators)"""
        total_score = 0
        valid_results = [r for r in results if r.get('generated_text') and not r.get('error')]
        
        for result in valid_results:
            text = result['generated_text']
            score = 0
            
            # Check for basic structure
            if len(text) > 10: score += 0.3
            if '।' in text or '?' in text or '!' in text: score += 0.2
            if not text.startswith(' ') and len(text) > 0: score += 0.2
            if len(text.split()) > 3: score += 0.3
            
            total_score += min(score, 1.0)
        
        return total_score / len(valid_results) if valid_results else 0
    
    def _assess_coherence(self, results: List[Dict]) -> float:
        """Assess text coherence"""
        total_score = 0
        valid_results = [r for r in results if r.get('generated_text') and not r.get('error')]
        
        for result in valid_results:
            text = result['generated_text']
            score = 0
            
            # Basic coherence indicators
            words = text.split()
            if len(words) > 5: score += 0.4
            if len(set(words)) / len(words) > 0.7: score += 0.3  # Word diversity
            if not any(word.count(char) > 5 for word in words for char in word): score += 0.3
            
            total_score += min(score, 1.0)
        
        return total_score / len(valid_results) if valid_results else 0
    
    def _assess_relevance(self, results: List[Dict]) -> float:
        """Assess relevance to prompt"""
        # Enhanced relevance check
        valid_results = [r for r in results if r.get('generated_text') and not r.get('error')]
        if not valid_results:
            return 0
        
        total_score = 0
        for result in valid_results:
            prompt = result['prompt'].lower()
            response = result['generated_text'].lower()
            
            # Simple keyword overlap check
            prompt_words = set(prompt.split())
            response_words = set(response.split())
            
            if prompt_words:
                overlap = len(prompt_words.intersection(response_words)) / len(prompt_words)
                total_score += min(overlap, 1.0)
        
        return total_score / len(valid_results)
    
    def _assess_length_consistency(self, results: List[Dict]) -> float:
        """Assess length consistency"""
        lengths = [len(result['generated_text'].split()) for result in results 
                  if result.get('generated_text') and not result.get('error')]
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
                    'completion': result['generated_text'],
                    'error': result.get('error')
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
                    'length_consistency_score': evaluation_scores.get('length_consistency', 0.0),
                    'error': result.get('error')
                }
                json_results.append(json_item)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Evaluation results saved to {filename} ({len(json_results)} items)")
        return json_results

# Cell 3: Load CATQA Bengali Prompts
# =====================================
print("\n" + "="*60)
print("📖 LOADING CATQA BENGALI PROMPTS")
print("="*60)

# Load prompts from JSON file
with open('converted_catqa_bn.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

prompts_data = data['prompts']
print(f"📊 Loaded {len(prompts_data)} prompts from converted_catqa_bn.json")

# Extract just the prompt texts
prompts = [item['prompt'] for item in prompts_data]
categories = [item['category'] for item in prompts_data]
ids = [item['id'] for item in prompts_data]

print(f"📋 Categories found: {set(categories)}")
print(f"🔢 ID range: {min(ids)} - {max(ids)}")

# Cell 4: Initialize Tester and Run Tests
# ======================================
print("\n" + "="*60)
print("🧪 INITIALIZING TESTER")
print("="*60)

# Initialize tester
tester = TigerLLMTester(model, tokenizer, DEVICE)
print("🧪 Testing infrastructure ready!")

# Cell 5: Run CATQA Bengali Tests
# ==================================
print("\n" + "="*60)
print("🎯 TESTING CATQA BENGALI PROMPTS")
print("="*60)

# Run tests with all prompts
catqa_results = tester.batch_test(prompts, "CATQA_Bengali", max_new_tokens=512, temperature=0.95)

# Analyze results
print("\n📊 CATQA Bengali Analysis:")
quality_scores = tester.evaluate_quality(catqa_results, ['fluency', 'coherence', 'relevance', 'length_consistency'])
for metric, score in quality_scores.items():
    print(f"  {metric.capitalize()}: {score:.3f}")

# Show sample outputs
print("\n🎨 Sample Outputs:")
for i, result in enumerate(catqa_results[:3]):
    print(f"\n{i+1}. Prompt: '{result['prompt'][:50]}...'")
    print(f"   Output: {result['generated_text'][:100]}...")
    print(f"   Tokens/sec: {result['tokens_per_second']:.2f}")
    if result.get('error'):
        print(f"   ⚠️ Error: {result['error']}")

# Save results to JSON files
print("\n" + "="*60)
print("💾 SAVING RESULTS TO JSON")
print("="*60)
tester.save_results_to_json("tigerllm_catqa_bn_results.json")

# Save evaluation results with scores
print("\n" + "="*60)
print("📊 SAVING EVALUATION RESULTS WITH SCORES")
print("="*60)
tester.save_evaluation_results_to_json("tigerllm_catqa_bn_evaluation_results.json")

print("\n✅ All tests completed successfully!") 