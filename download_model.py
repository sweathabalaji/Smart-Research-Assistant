from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

model_name = "distilbert-base-cased-distilled-squad"
save_directory = "./models/distilbert-base-cased-distilled-squad"

# Download and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_directory)

# Download and save the model with PyTorch state dict
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
torch.save(model.state_dict(), f"{save_directory}/pytorch_model.bin")  # manually save

# Save config (needed for loading)
model.config.to_json_file(f"{save_directory}/config.json")

print("✅ Model and tokenizer saved successfully!")
