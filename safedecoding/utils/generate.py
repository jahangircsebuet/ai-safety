import torch
import numpy as np

def generate(model, tokenizer, input_ids, gen_config=None):
    if gen_config is None:
        gen_config = model.generation_config
        gen_config.max_new_tokens = 32
        
    
    # 📦 What is .unsqueeze(0)?
    # In PyTorch, .unsqueeze(dim) adds a new axis (dimension) at the specified position.
    # .unsqueeze(0) adds a batch dimension at the start (dim 0).
    # This is important because HuggingFace models expect input in the format: [batch_size, sequence_length]
    
    # 🧪 Example
    # Suppose you have a prompt:
    # prompt = "How can I break into a house?"
    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    # print(input_ids.shape)
    # ✅ Output: torch.Size([1, 10])

    # This already includes a batch size of 1 — tokenizer(..., return_tensors="pt") returns batched input.
    # But suppose you directly encoded a prompt without batching:
    # input_ids = tokenizer.encode(prompt)  # returns list, no batch
    # input_ids = torch.tensor(input_ids)
    # print(input_ids.shape)
    # 🧨 Output: torch.Size([10])
    # Now the shape is [sequence_length], but the model requires [batch_size, sequence_length].
    input_ids = input_ids.to(model.device).unsqueeze(0)
    input_len = input_ids.shape[1]

    # 🎯 Create Attention Mask
    # Creates an attention mask filled with 1s (no padding).
    # Tells the model to pay attention to all tokens in the input prompt.
    attn_masks = torch.ones_like(input_ids).to(model.device)
    
    # Runs sampling or decoding from prompt
    output_ids = model.generate(input_ids=input_ids, 
                                attention_mask=attn_masks, 
                                generation_config=gen_config,
                                pad_token_id=tokenizer.pad_token_id)[0]

    # print(f"decoded output: [{tokenizer.decode(output_ids)}]")

    # ✂️ Return Only the Generated Tokens
    # Slices off the original prompt (input_len) and returns only the generated portion (i.e., continuation).
    # This keeps output clean — helpful for evaluation or filtering (like checking if it's a rejection).
    return output_ids[input_len:], output_ids