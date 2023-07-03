import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device("cpu")

tokenizer = GPT2Tokenizer.from_pretrained("stanford-crfm/BioMedLM")

model = GPT2LMHeadModel.from_pretrained("stanford-crfm/BioMedLM").to(device)

input_ids = tokenizer.encode(
    "A beginner-friendly explanation of a viruses is: ", return_tensors="pt"
).to(device)

sample_output = model.generate(input_ids, do_sample=True, max_length=100, top_k=50)

print("Output:\n" + 100 * "-")
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))