import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from transformers import BertTokenizer, BertModel
import os
import numpy as np

# Disable symlink warning
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Load the dataset
data = pd.read_csv('../data.csv')

print(data['label'].value_counts())

# Split the dataset into training and testing sets
X = data['sentence']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize the BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
def get_bert_embeddings(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Get BERT embeddings for all sentences in the training set
X_train_bert = []
for sentence in X_train:
    embedding = get_bert_embeddings(sentence)
    X_train_bert.append(embedding)

X_train_bert = np.vstack(X_train_bert)

# Get BERT embeddings for all sentences in the test set
X_test_bert = []
for sentence in X_test:
    embedding = get_bert_embeddings(sentence)
    X_test_bert.append(embedding)

X_test_bert = np.vstack(X_test_bert)

# Combine TF-IDF and BERT embeddings
X_train_combined = np.hstack((X_train_tfidf.toarray(), X_train_bert))
X_test_combined = np.hstack((X_test_tfidf.toarray(), X_test_bert))

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_combined, y_train)

# Save the trained model, vectorizer, and BERT model
joblib.dump(model, '../model/model.pkl')
joblib.dump(vectorizer, '../model/vectorizer.pkl')
joblib.dump(tokenizer, '../model/tokenizer.pkl')
joblib.dump(bert_model, '../model/bert_model.pkl')

# Print the classification report (optional)
y_pred = model.predict(X_test_combined)
print(classification_report(y_test, y_pred))
