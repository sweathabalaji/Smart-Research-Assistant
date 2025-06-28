from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/bart-large-cnn"
model_path = "./models/facebook-bart-large-cnn"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

print("✅ Summarizer model saved to:", model_path)
