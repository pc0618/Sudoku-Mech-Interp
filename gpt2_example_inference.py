# Import necessary libraries
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset

# Check if CUDA is available and set the device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load GPT-2 small model and tokenizer from Hugging Face
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Move the model to the selected device (GPU or CPU)
model = model.to(device)

# Load an example dataset (e.g., 'wikitext' dataset) for inference
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
example_text = dataset[4]["text"]
print(dataset[4]["text"])
# Tokenize the input text and handle empty strings
inputs = tokenizer(example_text, return_tensors="pt", truncation=True, max_length=512)

# Check for empty input tensor
if inputs["input_ids"].size(1) == 0:
    raise ValueError("The tokenized input is empty. Please check the input text or tokenization.")

inputs = inputs.to(device)

# Set pad_token_id to eos_token_id if pad_token_id is not defined for GPT-2
# GPT-2 doesn't have a pad_token by default, so we use the eos_token as a workaround
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss

# Get the logits (prediction scores)
logits = outputs.logits

# Convert logits to predicted token ids by taking the argmax (most likely token) for each position
# The last dimension of logits corresponds to the vocabulary size
predicted_token_ids = torch.argmax(logits, dim=-1)

# Convert the predicted token ids to human-readable text
generated_text = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
print(generated_text)
# Compute perplexity
perplexity = torch.exp(loss).item()
print(f"Perplexity: {perplexity}")