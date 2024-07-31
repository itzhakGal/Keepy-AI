import joblib
import os
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the model directory relative to the current script
model_dir = os.path.abspath(os.path.join(script_dir, '..', 'model'))

# Define paths to the files in the 'model' directory
model_path = os.path.join(model_dir, 'model.pkl')
vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')
bert_model_path = os.path.join(model_dir, 'bert_model.pkl')

# Load the trained model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
tokenizer = joblib.load(tokenizer_path)
bert_model = joblib.load(bert_model_path)


# Function to get BERT embeddings
def get_bert_embeddings(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()


def classify_sentence(sentence):
    # Convert sentence to TF-IDF features
    text_tfidf = vectorizer.transform([sentence])

    # Get BERT embeddings
    text_bert = get_bert_embeddings(sentence)

    # Combine TF-IDF and BERT embeddings
    combined_features = np.hstack((text_tfidf.toarray(), text_bert))

    # Predict using the combined features
    prediction = model.predict(combined_features)
    return prediction[0]


def main():
    print("Enter sentences to classify them as 'appropriate' or 'inappropriate'. Type 'exit' to quit.")
    while True:
        sentence = input("Enter a sentence: ")
        if sentence.lower() == 'exit':
            break
        label = classify_sentence(sentence)
        print(f"The sentence is classified as: {label}")


if __name__ == "__main__":
    main()
