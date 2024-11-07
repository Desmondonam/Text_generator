import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model and tokenizer
@st.cache_resource
def load_model():
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Text generation function
def generate_text(prompt, max_length=50, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        temperature=temperature,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit app layout
st.title("GPT-2 Text Generator")
st.write("Enter a prompt to generate text using GPT-2.")

# User input for prompt
prompt = st.text_input("Enter a prompt", "Once upon a time")

# Slider for controlling text generation length
max_length = st.slider("Select max length of generated text", 10, 100, 50)

# Temperature control
temperature = st.slider("Select temperature (creativity)", 0.1, 1.5, 0.7)

# Generate button
if st.button("Generate Text"):
    with st.spinner("Generating..."):
        generated_text = generate_text(prompt, max_length, temperature)
    st.write("Generated Text:", generated_text)
