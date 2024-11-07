from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Step 1: Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # Using GPT-2 small; you can try "gpt2-medium" for a larger model
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Step 2: Function for generating text based on a prompt
def generate_text(prompt, max_length=50, temperature=0.7):
    # Encode the prompt and return tensor
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate text
    output = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        temperature=temperature,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id  # Explicitly set pad_token_id to eos_token_id
    )
    
    # Decode and return the text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text
