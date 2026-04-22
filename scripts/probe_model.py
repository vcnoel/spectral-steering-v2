from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_id = "google/gemma-4-E2B-it"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print(f"BOS ID: {tokenizer.bos_token_id}, Token: {tokenizer.bos_token}")
print(f"EOS ID: {tokenizer.eos_token_id}, Token: {tokenizer.eos_token}")
print(f"PAD ID: {tokenizer.pad_token_id}, Token: {tokenizer.pad_token}")

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto", trust_remote_code=True)

text = "The capital of France is Paris."
inputs = tokenizer(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss.item()
    ppl = torch.exp(torch.tensor(loss)).item()
print(f"Simple sentence loss: {loss:.4f}, PPL: {ppl:.4f}")

# Check with BOS manually
inputs_bos = tokenizer(tokenizer.bos_token + text, return_tensors="pt").to(model.device) if tokenizer.bos_token else inputs
with torch.no_grad():
    outputs_bos = model(**inputs_bos, labels=inputs_bos["input_ids"])
    loss_bos = outputs_bos.loss.item()
    ppl_bos = torch.exp(torch.tensor(loss_bos)).item()
print(f"Simple sentence (with BOS) loss: {loss_bos:.4f}, PPL: {ppl_bos:.4f}")
