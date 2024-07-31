import joblib
import numpy as np
from config import Config


class SentenceClassifier:
    def __init__(self):
        self.model = joblib.load(Config.MODEL_PATH)
        self.vectorizer = joblib.load(Config.VECTORIZER_PATH)
        self.tokenizer = joblib.load(Config.TOKENIZER_PATH)
        self.bert_model = joblib.load(Config.BERT_MODEL_PATH)

    def get_bert_embeddings(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    def classify_sentence(self, sentence):
        text_tfidf = self.vectorizer.transform([sentence])
        text_bert = self.get_bert_embeddings(sentence)
        combined_features = np.hstack((text_tfidf.toarray(), text_bert))
        prediction = self.model.predict(combined_features)
        return prediction[0]
