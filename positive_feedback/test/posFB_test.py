import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

# Load the model and tokenizer
model_path = '../train/positive_model'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Sample sentences to test
test_sentences = [
    "you are the best.",
    "you are ok.",
    "you are champion.",
    "You are working on this task.",
    "I’m excited to see you learn something new."
]

# Tokenize the test sentences
inputs = tokenizer(test_sentences, return_tensors='pt', truncation=True, padding=True, max_length=512)

# Get model predictions
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)

# Convert predictions to labels
label_mapping = {0: "neutral", 1: "positive"}  # Adjust this based on how your labels are encoded
predicted_labels = [label_mapping[pred.item()] for pred in predictions]

# Print results
for sentence, label in zip(test_sentences, predicted_labels):
    print(f"Sentence: \"{sentence}\" is classified as: \"{label}\"")
