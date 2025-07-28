# Pangea-7B Multilingual Model Tester
# ===================================
# Supports 39 languages including Bangla
# Based on: https://huggingface.co/neulab/Pangea-7B

import json
import os
import time
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image

class PangeaTester:
    def __init__(self, model, tokenizer, device, image_processor=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.image_processor = image_processor
        self.results = {}
        self.test_start_time = datetime.now()
        
    def generate_text(self, prompt: str, max_new_tokens: int = 512,
                     temperature: float = 0.7, top_p: float = 0.9, 
                     do_sample: bool = True, return_metrics: bool = False,
                     image_path: Optional[str] = None) -> Dict:
        """Enhanced text generation with metrics for Pangea-7B"""
        start_time = time.time()
        
        try:
            # Handle multimodal input if image is provided
            # if image_path and self.image_processor:
            #     return self._generate_multimodal(prompt, image_path, max_new_tokens, temperature, top_p, do_sample, start_time)
            # else:
            return self._generate_text_only(prompt, max_new_tokens, temperature, top_p, do_sample, start_time)
        except Exception as e:
            print(f"⚠️ Generation error: {e}")
            return self._create_error_result(prompt, str(e), start_time)
    
    def _generate_text_only(self, prompt: str, max_new_tokens: int, temperature: float, 
                           top_p: float, do_sample: bool, start_time: float) -> Dict:
        """Generate text-only response"""
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                attention_mask=inputs.attention_mask
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_text = generated_text[len(prompt):].strip()
        
        return self._create_result(prompt, new_text, generated_text, inputs, outputs, start_time, temperature, top_p)
    
    def _generate_multimodal(self, prompt: str, image_path: str, max_new_tokens: int, 
                            temperature: float, top_p: float, do_sample: bool, start_time: float) -> Dict:
        """Generate multimodal response with image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            image_tensor = image_tensor.half().to(self.device)
            
            # Prepare multimodal prompt
            multimodal_prompt = f"<image>\n{prompt}"
            
            # Tokenize with image
            inputs = self.tokenizer(multimodal_prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to(self.device)
            
            # Generate with image
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    images=image_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=inputs.attention_mask
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_text = generated_text[len(multimodal_prompt):].strip()
            
            return self._create_result(prompt, new_text, generated_text, inputs, outputs, start_time, temperature, top_p, image_path)
            
        except Exception as e:
            print(f"⚠️ Multimodal generation error: {e}")
            return self._create_error_result(prompt, f"Multimodal error: {str(e)}", start_time, image_path)
    
    def _create_result(self, prompt: str, new_text: str, full_text: str, inputs, outputs, 
                      start_time: float, temperature: float, top_p: float, image_path: str = None) -> Dict:
        """Create standardized result dictionary"""
        generation_time = time.time() - start_time
        input_tokens = len(inputs.input_ids[0])
        output_tokens = len(outputs[0]) - input_tokens
        tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
        
        result = {
            'prompt': prompt,
            'full_text': full_text,
            'generated_text': new_text,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'generation_time': generation_time,
            'tokens_per_second': tokens_per_second,
            'temperature': temperature,
            'top_p': top_p,
            'has_image': image_path is not None,
            'image_path': image_path
        }
        
        return result
    
    def _create_error_result(self, prompt: str, error_msg: str, start_time: float, image_path: str = None) -> Dict:
        """Create error result when generation fails"""
        generation_time = time.time() - start_time
        
        return {
            'prompt': prompt,
            'full_text': '',
            'generated_text': '',
            'input_tokens': 0,
            'output_tokens': 0,
            'generation_time': generation_time,
            'tokens_per_second': 0,
            'temperature': 0,
            'top_p': 0,
            'has_image': image_path is not None,
            'image_path': image_path,
            'error': error_msg
        }
    
    def batch_test(self, prompts: List[str], test_name: str, image_paths: Optional[List[str]] = None, **kwargs) -> List[Dict]:
        """Run batch testing with progress tracking"""
        results = []
        print(f"🧪 Running {test_name} ({len(prompts)} tests)...")
        
        for i, prompt in enumerate(prompts):
            print(f"  Progress: {i+1}/{len(prompts)}", end="\r")
            
            # Handle image if provided
            image_path = None
            if image_paths and i < len(image_paths):
                image_path = image_paths[i]
            
            result = self.generate_text(prompt, image_path=image_path, **kwargs)
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
            elif criterion == 'multilingual_quality':
                scores[criterion] = self._assess_multilingual_quality(results)
        
        return scores
    
    def _assess_fluency(self, results: List[Dict]) -> float:
        """Assess text fluency for multilingual content"""
        total_score = 0
        valid_results = [r for r in results if r.get('generated_text') and not r.get('error')]
        
        for result in valid_results:
            text = result['generated_text']
            score = 0
            
            # Check for basic structure (works for multiple languages)
            if len(text) > 10: score += 0.3
            if any(punct in text for punct in ['।', '?', '!', '.', '؟', '！', '？']): score += 0.2
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
        # Enhanced relevance check for multilingual content
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
    
    def _assess_multilingual_quality(self, results: List[Dict]) -> float:
        """Assess multilingual generation quality"""
        valid_results = [r for r in results if r.get('generated_text') and not r.get('error')]
        if not valid_results:
            return 0
        
        total_score = 0
        for result in valid_results:
            text = result['generated_text']
            score = 0
            
            # Check for multilingual indicators
            # Bangla characters
            if any('\u0980' <= char <= '\u09FF' for char in text): score += 0.3
            # English characters
            if any('a' <= char.lower() <= 'z' for char in text): score += 0.2
            # Proper sentence structure
            if len(text.split()) > 5: score += 0.3
            # No excessive repetition
            words = text.split()
            if len(set(words)) / len(words) > 0.6: score += 0.2
            
            total_score += min(score, 1.0)
        
        return total_score / len(valid_results)
    
    def save_results_to_json(self, filename: str = "pangea_results.json"):
        """Save all test results to JSON file with test_name, prompt, completion structure"""
        json_results = []
        
        for test_name, test_results in self.results.items():
            for result in test_results:
                json_item = {
                    'test_name': test_name,
                    'prompt': result['prompt'],
                    'completion': result['generated_text'],
                    'has_image': result.get('has_image', False),
                    'image_path': result.get('image_path'),
                    'error': result.get('error')
                }
                json_results.append(json_item)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Results saved to {filename} ({len(json_results)} items)")
        return json_results
    
    def save_evaluation_results_to_json(self, filename: str = "pangea_evaluation_results.json"):
        """Save all test results with evaluation scores to JSON file"""
        json_results = []
        
        for test_name, test_results in self.results.items():
            # Calculate evaluation scores for this test
            evaluation_scores = self.evaluate_quality(test_results, 
                ['fluency', 'coherence', 'relevance', 'length_consistency', 'multilingual_quality'])
            
            for result in test_results:
                json_item = {
                    'test_name': test_name,
                    'prompt': result['prompt'],
                    'completion': result['generated_text'],
                    'fluency_score': evaluation_scores.get('fluency', 0.0),
                    'coherence_score': evaluation_scores.get('coherence', 0.0),
                    'relevance_score': evaluation_scores.get('relevance', 0.0),
                    'length_consistency_score': evaluation_scores.get('length_consistency', 0.0),
                    'multilingual_quality_score': evaluation_scores.get('multilingual_quality', 0.0),
                    'has_image': result.get('has_image', False),
                    'image_path': result.get('image_path'),
                    'error': result.get('error')
                }
                json_results.append(json_item)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Evaluation results saved to {filename} ({len(json_results)} items)")
        return json_results
    
    def get_model_info(self) -> Dict:
        """Get model information and statistics"""
        param_count = sum(p.numel() for p in self.model.parameters())
        memory_usage = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        
        return {
            'model_name': 'Pangea-7B',
            'parameters': param_count,
            'parameters_billions': param_count / 1e9,
            'memory_usage_gb': memory_usage,
            'device': str(self.device),
            'supports_multimodal': self.image_processor is not None,
            'supported_languages': 39,
            'languages': [
                'Amharic', 'Arabic', 'Bengali', 'Bulgarian', 'Czech', 'German', 'Greek', 
                'English', 'Spanish', 'Persian', 'French', 'Irish', 'Hindi', 'Indonesian', 
                'Nigerian', 'Italian', 'Hebrew', 'Japanese', 'Korean', 'Dutch', 'Mongolian', 
                'Malay', 'Norwegian', 'Polish', 'Portuguese', 'Brazilian', 'Romanian', 
                'Russian', 'Sinhala', 'Swahili', 'Thai', 'Turkish', 'Ukrainian', 'Pakistani', 
                'Vietnamese', 'Chinese', 'Taiwanese'
            ]
        } 