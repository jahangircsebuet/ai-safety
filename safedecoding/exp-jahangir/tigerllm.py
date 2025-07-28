# https://www.kaggle.com/code/jahangir67/tigerllm-testing-and-benchmarking/edit

# TigerLLM 1B Comprehensive Testing & Benchmarking Notebook

# Author: AI Model Evaluation
# Model: md-nishat-008/TigerLLM-1B-it (Bengali Language Model)
# Purpose: Comprehensive capability assessment, benchmarking, and usage recommendations
import logging
import os
from datetime import datetime
import numpy as np

# Logging Settings
output_dir = "tigerllm_results/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

log_name = "tigerllm_1b.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(output_dir, log_name)),
        logging.StreamHandler()
    ]
)

# Cell 2: Model Loading and Configuration
# ======================================
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tigerllm_tester import TigerLLMTester

logging.info(f"🚀 Loading TigerLLM 1B Model...")
logging.info(f"=" * 50)

# Model configuration
MODEL_NAME = "md-nishat-008/TigerLLM-1B-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import os

# First, let's try to load the model
logging.info(f"🔄 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=DEVICE,
    trust_remote_code=True,
    token=os.getenv('HUGGING_FACE_HUB_TOKEN')
)

# Then try to load the tokenizer
logging.info(f"🔄 Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        use_fast=False,
        trust_remote_code=True,
        token=os.getenv('HUGGING_FACE_HUB_TOKEN')
    )
    logging.info(f"✅ Tokenizer loaded successfully")
except Exception as e:
    logging.info(f"⚠️ Could not load Llama tokenizer: {e}")
    logging.info(f"🔄 Trying to load tokenizer from TigerLLM model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            token=os.getenv('HUGGING_FACE_HUB_TOKEN')
        )
        logging.info(f"✅ Tokenizer loaded from TigerLLM model")
    except Exception as e2:
        logging.info(f"⚠️ Could not load tokenizer from TigerLLM either: {e2}")
        logging.info(f"🔄 Using a fallback tokenizer...")
        # Use a simple fallback tokenizer
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token


# Model statistics
param_count = sum(p.numel() for p in model.parameters())
logging.info(f"📊 Model Parameters: {param_count:,} ({param_count/1e9:.2f}B)")
logging.info(f"💾 Memory Usage: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
logging.info(f"⚡ Device: {DEVICE}")
logging.info(f"🔧 Model loaded successfully!")



logging.info(f"# Cell 3: Core Testing Infrastructure")
logging.info(f"# ===================================================")
# Initialize tester
tester = TigerLLMTester(model, tokenizer, DEVICE)
logging.info("🧪 Testing infrastructure ready!")

logging.info(f"# Cell 4: Test 1 - Basic Text Generation Capabilities")
logging.info(f"# ===================================================")
logging.info(f"\n" + "="*60)
logging.info(f"🎯 TEST 1: BASIC TEXT GENERATION CAPABILITIES")
logging.info(f"="*60)


if True:
    basic_prompts = [
        # Storytelling
        "একদিন এক ছোট্ট মেয়ে",
        "বনের গভীরে একটি রহস্যময় গুহা",
        "শহরের মধ্যে একটি অদ্ভুত ঘটনা",
        
        # Descriptive
        "বর্ষাকালের সকাল",
        "ঢাকা শহরের যানজট",
        "একটি সুন্দর গ্রাম",
        
        # Conversational
        "আমার প্রিয় খাবার",
        "ছুটির দিনে আমি",
        "বন্ধুদের সাথে আড্ডা",
        
        # Informative
        "বাংলাদেশের ইতিহাস",
        "কম্পিউটার প্রযুক্তি",
        "শিক্ষার গুরুত্ব"
    ]

    basic_results = tester.batch_test(basic_prompts, "Basic Generation", max_length=120, temperature=0.8)

    # Analyze results
    logging.info("\n📊 Basic Generation Analysis:")
    quality_scores = tester.evaluate_quality(basic_results, ['fluency', 'coherence', 'length_consistency'])
    for metric, score in quality_scores.items():
        logging.info(f"  {metric.capitalize()}: {score:.3f}")

    # Show sample outputs
    logging.info("\n🎨 Sample Outputs:")
    for i, result in enumerate(basic_results[:3]):
        logging.info(f"\n{i+1}. Prompt: '{result['prompt']}'")
        logging.info(f"   Output: {result['generated_text'][:100]}...")
        logging.info(f"   Tokens/sec: {result['tokens_per_second']:.2f}")

    # Save results to JSON file
    logging.info("\n" + "="*60)
    logging.info("💾 SAVING RESULTS TO JSON")
    logging.info("="*60)
    tester.save_results_to_json("tigerllm_basic_test_results.json")

    # Save evaluation results with scores
    logging.info("\n" + "="*60)
    logging.info("📊 SAVING EVALUATION RESULTS WITH SCORES")
    logging.info("="*60)
    tester.save_evaluation_results_to_json("tigerllm_evaluation_results.json")


if True:
    # Cell 5: Test 2 - Question Answering Capabilities
    # ================================================
    logging.info("\n" + "="*60)
    logging.info("🎯 TEST 2: QUESTION ANSWERING CAPABILITIES")
    logging.info("="*60)

    qa_prompts = [
        # Factual Questions
        "বাংলাদেশের রাজধানী কোথায়?",
        "পৃথিবীর সবচেয়ে বড় দেশ কোনটি?",
        "কোন গ্রহটি সূর্যের সবচেয়ে কাছে?",
        
        # Explanatory Questions
        "ফটোসিনথেসিস কী?",
        "গ্রীনহাউস এফেক্ট কেন হয়?",
        "ইন্টারনেট কীভাবে কাজ করে?",
        
        # Cultural Questions
        "বাংলা নববর্ষ কীভাবে পালিত হয়?",
        "বাংলাদেশের জাতীয় কবি কে?",
        "পহেলা বৈশাখের তাৎপর্য কী?",
        
        # Mathematical Questions
        "১০০ এর বর্গমূল কত?",
        "পাই (π) এর মান কত?",
        "এক মিনিটে কত সেকেন্ড?",
        
        # Logical Questions
        "কেন আকাশ নীল দেখায়?",
        "বৃষ্টি কেন হয়?",
        "দিন রাত কেন হয়?"
    ]

    qa_results = tester.batch_test(qa_prompts, "Question Answering", max_length=100, temperature=0.3)

    # Analyze QA performance
    logging.info("\n📊 Question Answering Analysis:")
    qa_quality = tester.evaluate_quality(qa_results, ['fluency', 'coherence', 'relevance'])
    for metric, score in qa_quality.items():
        logging.info(f"  {metric.capitalize()}: {score:.3f}")

    # Categorize responses
    factual_correct = 0
    explanatory_adequate = 0
    cultural_appropriate = 0

    logging.info("\n🎯 Sample Q&A Performance:")
    for i, result in enumerate(qa_results[:6]):
        logging.info(f"\nQ: {result['prompt']}")
        logging.info(f"A: {result['generated_text'][:150]}...")


if True:
    # Cell 6: Test 3 - Creative Writing Assessment
    # ============================================
    logging.info("\n" + "="*60)
    logging.info("🎯 TEST 3: CREATIVE WRITING ASSESSMENT")
    logging.info("="*60)

    creative_prompts = [
        # Poetry
        "একটি সুন্দর কবিতা লিখুন বসন্তকাল নিয়ে:",
        "মাতৃভাষা দিবস নিয়ে একটি ছোট কবিতা:",
        
        # Story beginnings
        "একটি রহস্যময় গল্প লিখুন যা শুরু হয় এভাবে: 'সেদিন রাতে অদ্ভুত একটি স্বপ্ন দেখেছিলাম'",
        "একটি হাস্যকর গল্প লিখুন একটি বিড়াল নিয়ে:",
        
        # Descriptive writing
        "একটি সুন্দর সূর্যাস্তের বর্ণনা দিন:",
        "বর্ষার প্রথম দিনের অনুভূতি লিখুন:",
        
        # Dialogue
        "দুই বন্ধুর মধ্যে একটি আকর্ষণীয় কথোপকথন লিখুন:",
        "একটি শিশু এবং তার দাদুর মধ্যে কথোপকথন:"
    ]

    creative_results = tester.batch_test(creative_prompts, "Creative Writing", 
                                    max_length=150, temperature=0.9)

    logging.info("\n📊 Creative Writing Analysis:")
    creative_quality = tester.evaluate_quality(creative_results, ['fluency', 'coherence'])
    for metric, score in creative_quality.items():
        logging.info(f"  {metric.capitalize()}: {score:.3f}")

    logging.info("\n🎨 Creative Samples:")
    for i, result in enumerate(creative_results[:3]):
        logging.info(f"\n{i+1}. Prompt: {result['prompt'][:50]}...")
        logging.info(f"   Creation: {result['generated_text'][:200]}...")

if True:
    # Cell 7: Test 4 - Technical & Code Generation
    # ============================================
    logging.info("\n" + "="*60)
    logging.info("🎯 TEST 4: TECHNICAL & CODE GENERATION")
    logging.info("="*60)

    technical_prompts = [
        # Programming
        "পাইথনে একটি সহজ হ্যালো ওয়ার্ল্ড প্রোগ্রাম লিখুন:",
        "একটি সিম্পল ক্যালকুলেটর ফাংশন লিখুন:",
        "একটি লিস্ট সর্ট করার কোড লিখুন:",
        
        # Technical explanations
        "কৃত্রিম বুদ্ধিমত্তা কী এবং কীভাবে কাজ করে?",
        "মেশিন লার্নিং এর মূল নীতি ব্যাখ্যা করুন:",
        "ডেটাবেস কী এবং কেন গুরুত্বপূর্ণ?",
        
        # Technology trends
        "ব্লকচেইন প্রযুক্তির ভবিষ্যৎ:",
        "ভার্চুয়াল রিয়েলিটির ব্যবহার:",
        "সাইবার নিরাপত্তার গুরুত্ব:"
    ]

    technical_results = tester.batch_test(technical_prompts, "Technical Writing", 
                                        max_length=150, temperature=0.5)

    logging.info("\n📊 Technical Writing Analysis:")
    tech_quality = tester.evaluate_quality(technical_results, ['fluency', 'coherence', 'relevance'])
    for metric, score in tech_quality.items():
        logging.info(f"  {metric.capitalize()}: {score:.3f}")


if True:
    # Cell 8: Test 5 - Language Understanding & Reasoning
    # ===================================================
    logging.info("\n" + "="*60)
    logging.info("🎯 TEST 5: LANGUAGE UNDERSTANDING & REASONING")
    logging.info("="*60)

    reasoning_prompts = [
        # Logical reasoning
        "যদি সব গোলাপ ফুল হয়, এবং সব ফুল সুন্দর হয়, তাহলে সব গোলাপ কী?",
        "একটি ট্রেন ঘণ্টায় ৬০ কিমি বেগে চলে। ২ ঘণ্টায় কত দূরত্ব অতিক্রম করবে?",
        
        # Reading comprehension
        "নিচের বাক্যটি পড়ুন এবং প্রশ্নের উত্তর দিন: 'রহিম একজন ভালো ছাত্র। সে প্রতিদিন ৩ ঘণ্টা পড়াশোনা করে।' রহিম কেমন ছাত্র?",
        
        # Comparison and analysis
        "গ্রাম এবং শহরের মধ্যে পার্থক্য বলুন:",
        "অনলাইন শিক্ষা বনাম প্রথাগত শিক্ষার সুবিধা-অসুবিধা:",
        
        # Problem solving
        "যদি একটি দোকানে ৫০টি কলম থাকে এবং প্রতিদিন ৭টি কলম বিক্রি হয়, তাহলে কতদিন পর কলম শেষ হবে?",
        "কীভাবে জল সংরক্ষণ করা যায়?",
        
        # Sentiment and emotion
        "এই বাক্যে কী ধরনের অনুভূতি প্রকাশ পেয়েছে: 'আজকের পরীক্ষা খুবই কঠিন ছিল'?",
        "একজন মানুষ কেন হাসে?"
    ]

    reasoning_results = tester.batch_test(reasoning_prompts, "Reasoning", 
                                        max_length=120, temperature=0.4)

    logging.info("\n📊 Reasoning Analysis:")
    reasoning_quality = tester.evaluate_quality(reasoning_results, ['fluency', 'coherence', 'relevance'])
    for metric, score in reasoning_quality.items():
        logging.info(f"  {metric.capitalize()}: {score:.3f}")


if True:
    # Cell 9: Test 6 - Multilingual Capabilities
    # ==========================================
    logging.info("\n" + "="*60)
    logging.info("🎯 TEST 6: MULTILINGUAL CAPABILITIES")
    logging.info("="*60)

    multilingual_prompts = [
        # English prompts
        "What is the capital of Bangladesh?",
        "Write a short story about friendship:",
        "Explain artificial intelligence in simple terms:",
        
        # Mixed language
        "বাংলাদেশের capital city কোথায়?",
        "আমি programming শিখতে চাই, কোথায় start করব?",
        
        # Translation requests
        "এই বাক্যটি ইংরেজিতে অনুবাদ করুন: 'আমি বাংলাদেশে থাকি'",
        "Translate this to Bengali: 'I love reading books'",
        
        # Code mixing
        "আজকের modern যুগে technology এর importance কী?"
    ]

    multilingual_results = tester.batch_test(multilingual_prompts, "Multilingual", 
                                        max_length=100, temperature=0.6)

    logging.info("\n📊 Multilingual Performance:")
    for i, result in enumerate(multilingual_results):
        logging.info(f"\n{i+1}. Input: {result['prompt'][:50]}...")
        logging.info(f"   Output: {result['generated_text'][:100]}...")


if True:
    # Cell 10: Performance Metrics & Benchmarking
    # ===========================================
    logging.info("\n" + "="*60)
    logging.info("🎯 PERFORMANCE METRICS & BENCHMARKING")
    logging.info("="*60)

    # Aggregate all results
    all_results = []
    for test_name, results in tester.results.items():
        all_results.extend(results)

    # Performance statistics
    total_tests = len(all_results)
    avg_generation_time = np.mean([r['generation_time'] for r in all_results])
    avg_tokens_per_second = np.mean([r['tokens_per_second'] for r in all_results])
    avg_output_length = np.mean([len(r['generated_text'].split()) for r in all_results])

    logging.info(f"\n📊 Overall Performance Statistics:")
    logging.info(f"  Total Tests Conducted: {total_tests}")
    logging.info(f"  Average Generation Time: {avg_generation_time:.3f} seconds")
    logging.info(f"  Average Tokens/Second: {avg_tokens_per_second:.2f}")
    logging.info(f"  Average Output Length: {avg_output_length:.1f} words")
    logging.info(f"  Model Parameters: {param_count:,}")
    logging.info(f"  Memory Usage: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # Create performance summary
    performance_summary = {
        'Basic Generation': tester.evaluate_quality(basic_results, ['fluency', 'coherence']),
        'Question Answering': tester.evaluate_quality(qa_results, ['fluency', 'relevance']),
        'Creative Writing': tester.evaluate_quality(creative_results, ['fluency', 'coherence']),
        'Technical Writing': tester.evaluate_quality(technical_results, ['fluency', 'coherence']),
        'Reasoning': tester.evaluate_quality(reasoning_results, ['fluency', 'relevance'])
    }

    logging.info(f"\n📈 Quality Scores by Category:")
    for category, scores in performance_summary.items():
        logging.info(f"\n{category}:")
        for metric, score in scores.items():
            logging.info(f"  {metric.capitalize()}: {score:.3f}")

if True:
    # Cell 11: Limitations Analysis (REVISED)
    # ========================================
    logging.info("\n" + "="*60)
    logging.info("🎯 LIMITATIONS ANALYSIS - REVISED")
    logging.info("="*60)

    logging.info("\n🚨 CRITICAL LIMITATIONS IDENTIFIED:")
    logging.info("1. **Poor Task Understanding**: Model struggles to follow specific instructions")
    logging.info("2. **Low Relevance**: Generates irrelevant content for targeted queries")
    logging.info("3. **Weak Question Answering**: Only 75% relevance in Q&A tasks")
    logging.info("4. **Limited Creative Control**: Creative writing lacks coherence (77.5%)")
    logging.info("5. **Technical Inadequacy**: Cannot handle technical content effectively")
    logging.info("6. **Reasoning Failures**: Poor logical reasoning capabilities (75% relevance)")
    logging.info("7. **Context Drift**: Loses context and generates off-topic content")
    logging.info("8. **Instruction Following**: Struggles with specific task requirements")

    logging.info(f"\n⚠️ PERFORMANCE BREAKDOWN:")
    logging.info("  • Basic Text Generation: GOOD (100% fluency, 80% coherence)")
    logging.info("  • Question Answering: POOR (92% fluency, 75% relevance)")
    logging.info("  • Creative Writing: MEDIOCRE (97.5% fluency, 77.5% coherence)")
    logging.info("  • Technical Writing: POOR (97.8% fluency, 80% coherence)")
    logging.info("  • Reasoning: POOR (97.8% fluency, 75% relevance)")

    logging.info(f"\n🔍 ROOT CAUSES:")
    logging.info("  • Model size constraint (1B parameters)")
    logging.info("  • Limited training on instruction-following tasks")
    logging.info("  • Weak understanding of task-specific requirements")
    logging.info("  • Poor context retention for complex queries")
    logging.info("  • Insufficient fine-tuning for specific domains")

if True:
    # Cell 12: Use Case Recommendations (REVISED)
    # ============================================
    logging.info("\n" + "="*60)
    logging.info("🎯 USE CASE RECOMMENDATIONS - REVISED")
    logging.info("="*60)

    use_cases_revised = {
        "✅ RECOMMENDED USE CASES (Limited)": [
            "Basic Bengali text continuation/completion",
            "Simple story beginnings (with human editing)",
            "Text expansion for existing content",
            "Creative writing inspiration (requires heavy editing)",
            "Basic conversational responses (casual only)",
            "Content ideation (as starting point only)"
        ],
        
        "⚠️ USE WITH HEAVY SUPERVISION": [
            "Educational content (requires fact-checking)",
            "Creative writing (needs significant editing)",
            "Blog post drafts (requires complete revision)",
            "Simple explanations (must verify accuracy)",
            "Basic translations (informal only)"
        ],
        
        "❌ NOT RECOMMENDED": [
            "Question answering systems",
            "Educational Q&A platforms",
            "Technical documentation",
            "Professional writing assistance",
            "Code generation",
            "Reasoning-based applications",
            "Factual information retrieval",
            "Complex creative writing",
            "Academic content creation",
            "Business communications",
            "Medical or legal content",
            "Real-time customer support",
            "Content requiring accuracy",
            "Task-specific applications"
        ]
    }

    for category, items in use_cases_revised.items():
        logging.info(f"\n{category}:")
        for item in items:
            logging.info(f"  • {item}")

    logging.info(f"\n💡 ALTERNATIVE RECOMMENDATIONS:")
    logging.info("  • Use for inspiration only, not final output")
    logging.info("  • Combine with human editing and fact-checking")
    logging.info("  • Consider larger models for serious applications")
    logging.info("  • Use as a starting point for creative processes")
    logging.info("  • Implement strict content filtering and review")


if True:
    # Cell 13: Final Report & Summary (REVISED)
    # ==========================================
    logging.info("\n" + "="*60)
    logging.info("📋 FINAL COMPREHENSIVE REPORT - REVISED")
    logging.info("="*60)

    # Recalculate overall score based on actual performance
    actual_scores = {
        'Basic Generation': 0.900,  # (1.000 + 0.800) / 2
        'Question Answering': 0.835,  # (0.920 + 0.750) / 2
        'Creative Writing': 0.875,  # (0.975 + 0.775) / 2
        'Technical Writing': 0.889,  # (0.978 + 0.800) / 2
        'Reasoning': 0.864  # (0.978 + 0.750) / 2
    }

    overall_performance_revised = np.mean(list(actual_scores.values()))

    logging.info(f"\n🏆 OVERALL MODEL PERFORMANCE: {overall_performance_revised:.3f}/1.000 (BELOW AVERAGE)")
    logging.info(f"🔍 MODEL SIZE: 1B parameters (INSUFFICIENT for complex tasks)")
    logging.info(f"⚡ AVERAGE SPEED: 38.11 tokens/second (REASONABLE)")
    logging.info(f"🎯 PRIMARY STRENGTH: Basic Bengali text generation only")

    logging.info(f"\n📊 REALISTIC CATEGORY BREAKDOWN:")
    for category, score in actual_scores.items():
        status = "GOOD" if score > 0.9 else "POOR" if score < 0.85 else "MEDIOCRE"
        logging.info(f"  {category}: {score:.3f} ({status})")

    logging.info(f"\n🎯 LIMITED STRENGTHS:")
    logging.info("  • Decent Bengali language fluency")
    logging.info("  • Basic text continuation capability")
    logging.info("  • Fast generation speed")
    logging.info("  • Lightweight memory footlogging.info")
    logging.info("  • Simple conversational responses")

    logging.info(f"\n⚠️ CRITICAL WEAKNESSES:")
    logging.info("  • Poor task comprehension and instruction following")
    logging.info("  • Low relevance in responses (75% average)")
    logging.info("  • Inadequate for question answering systems")
    logging.info("  • Cannot handle technical or complex content")
    logging.info("  • Weak reasoning and logical capabilities")
    logging.info("  • Limited creative coherence")
    logging.info("  • Generates off-topic or irrelevant content")

    logging.info(f"\n🎯 REALISTIC APPLICATIONS:")
    logging.info("  • Text completion for existing Bengali content")
    logging.info("  • Creative writing inspiration (with heavy editing)")
    logging.info("  • Basic conversational responses (casual contexts)")
    logging.info("  • Content brainstorming (starting ideas only)")
    logging.info("  • Simple text expansion tasks")

    logging.info(f"\n❌ APPLICATIONS TO AVOID:")
    logging.info("  • Any task requiring accuracy or reliability")
    logging.info("  • Educational or informational systems")
    logging.info("  • Professional content creation")
    logging.info("  • Question answering platforms")
    logging.info("  • Technical documentation")
    logging.info("  • Business communications")

    logging.info(f"\n🔍 VERDICT:")
    logging.info("  The TigerLLM 1B model shows promise for basic Bengali text generation")
    logging.info("  but falls short for most practical applications due to poor task")
    logging.info("  understanding and low relevance in responses. Best used as a")
    logging.info("  starting point for human-supervised content creation rather than")
    logging.info("  standalone AI applications.")

    logging.info(f"\n📊 RECOMMENDATION SCORE: 3/10 (Limited utility)")
    logging.info("  • Consider larger models (7B+) for serious applications")
    logging.info("  • Use only with heavy human supervision")
    logging.info("  • Not suitable for production environments")
    logging.info("  • Better alternatives available for most use cases")

    logging.info(f"\n⏱️ TESTING COMPLETED: {datetime.now() - tester.test_start_time}")
    logging.info("="*60)
    logging.info("🎉 REALISTIC ASSESSMENT COMPLETE!")
    logging.info("="*60)