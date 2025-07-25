# command 1: rejection dataset -> generate dataset for finetuning 
# (pass OpenAI API Key, 
# set lets_gen_finetune_dataset = True and 
# lets_finetune = False in the code,
# dataset will be stored in lora_modules directory
# CUDA_VISIBLE_DEVICES=2 - this for telling to use gpu device 2)
#CUDA_VISIBLE_DEVICES=2 python finetune.py --model_name vicuna --GPT_API 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

# command 2: finetune -> generate dataset for finetuning 
# (pass OpenAI API Key, but key will not be used, 
# set lets_gen_finetune_dataset = False and 
# lets_finetune = True in the code
# expert model will be stored in lora_modules directory
# CUDA_VISIBLE_DEVICES=2 - this for telling to use gpu device 2)
#CUDA_VISIBLE_DEVICES=2 python finetune.py --model_name vicuna --GPT_API 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

# command 3: ealuate -> evaluate vicuna model's generation applying safedecoding 
# (vicuna original and vicuna expert model will be used, 
# token distribution will be generated,
# AdvBench attacker (which is a set of unsafe queries) will be used,
# SafeDecoding defender (which utilizes original and expert model) will be used,
# disable_GPT_judge - this will tell not to call GPT to get harmfulness score
# CUDA_VISIBLE_DEVICES=2 - this for telling to use gpu device 2)
#CUDA_VISIBLE_DEVICES=2 python defense.py --model_name vicuna --attacker AdvBench --defender SafeDecoding --disable_GPT_judge

# command 4: 
#CUDA_VISIBLE_DEVICES=2 python defense.py --model_name vicuna --attacker [YOUR_ATTACKER_NAME] --defender [YOUR_DEFENDER_NAME] --GPT_API [YOUR_OPENAI_API]

# command 1: rejection dataset -> generate dataset for finetuning 
# (pass OpenAI API Key, 
# set lets_gen_finetune_dataset = True and 
# lets_finetune = False in the code,
# dataset will be stored in lora_modules directory
# CUDA_VISIBLE_DEVICES=2 - this for telling to use gpu device 2)
# CUDA_VISIBLE_DEVICES=2 python finetune_bangla.py --model_name bloom --GPT_API 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

CUDA_VISIBLE_DEVICES=2 python test_bloom.py --model_name bloom --GPT_API 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


