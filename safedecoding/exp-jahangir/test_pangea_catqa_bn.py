# Test Pangea-7B with CATQA Bengali prompts
# =============================================
# Based on: https://huggingface.co/neulab/Pangea-7B

import json
import os
import time
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from pangea_tester import PangeaTester

# Cell 1: Model Loading and Configuration
# ======================================
print("🚀 Loading Pangea-7B Multilingual Model...")
print("=" * 60)

# Model configuration
MODEL_NAME = "neulab/Pangea-7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
print("🔄 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=DEVICE,
    trust_remote_code=True,
    token=os.getenv('HUGGING_FACE_HUB_TOKEN')
)

print("🔄 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    token=os.getenv('HUGGING_FACE_HUB_TOKEN')
)

# Model statistics
param_count = sum(p.numel() for p in model.parameters())
print(f"📊 Model Parameters: {param_count:,} ({param_count/1e9:.2f}B)")
print(f"💾 Memory Usage: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"⚡ Device: {DEVICE}")
print(f"🔧 Model loaded successfully!")

# Cell 2: Initialize PangeaTester
# ===============================
print("\n" + "="*60)
print("🧪 INITIALIZING PANGEA TESTER")
print("="*60)

# Initialize tester (text-only mode for now)
tester = PangeaTester(model, tokenizer, DEVICE)
print("🧪 Pangea testing infrastructure ready!")

# Get model info
model_info = tester.get_model_info()
print(f"🌍 Supported Languages: {model_info['supported_languages']}")
print(f"🔤 Languages: {', '.join(model_info['languages'][:10])}...")

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

# Cell 4: Run CATQA Bengali Tests
# ==================================
print("\n" + "="*60)
print("🎯 TESTING PANGEA-7B WITH CATQA BENGALI PROMPTS")
print("="*60)

# Run tests with all prompts
print("🚀 Starting batch testing...")
catqa_results = tester.batch_test(prompts, "Pangea_CATQA_Bengali", max_new_tokens=512,temperature=0.95)

# Analyze results
print("\n📊 Pangea-7B CATQA Bengali Analysis:")
quality_scores = tester.evaluate_quality(catqa_results, 
    ['fluency', 'coherence', 'relevance', 'length_consistency', 'multilingual_quality'])

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

# Cell 5: Save Results
# ===================
print("\n" + "="*60)
print("💾 SAVING RESULTS TO JSON")
print("="*60)

# Save basic results
tester.save_results_to_json("pangea_catqa_bn_results.json")

# Save evaluation results with scores
print("\n" + "="*60)
print("📊 SAVING EVALUATION RESULTS WITH SCORES")
print("="*60)
tester.save_evaluation_results_to_json("pangea_catqa_bn_evaluation_results.json")

# Cell 6: Performance Summary
# ==========================
print("\n" + "="*60)
print("📈 PERFORMANCE SUMMARY")
print("="*60)

# Calculate performance metrics
total_tests = len(catqa_results)
successful_tests = len([r for r in catqa_results if not r.get('error')])
failed_tests = total_tests - successful_tests

avg_generation_time = np.mean([r['generation_time'] for r in catqa_results if not r.get('error')])
avg_tokens_per_sec = np.mean([r['tokens_per_second'] for r in catqa_results if not r.get('error')])

print(f"📊 Test Statistics:")
print(f"  Total Tests: {total_tests}")
print(f"  Successful: {successful_tests} ({successful_tests/total_tests*100:.1f}%)")
print(f"  Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
print(f"  Avg Generation Time: {avg_generation_time:.3f}s")
print(f"  Avg Tokens/sec: {avg_tokens_per_sec:.2f}")

print(f"\n🎯 Quality Scores:")
for metric, score in quality_scores.items():
    print(f"  {metric.replace('_', ' ').title()}: {score:.3f}")

print(f"\n🌍 Multilingual Performance:")
print(f"  Model: Pangea-7B ({model_info['parameters_billions']:.1f}B parameters)")
print(f"  Supported Languages: {model_info['supported_languages']}")
print(f"  Bangla Support: ✅ Included")

print("\n✅ All tests completed successfully!") 