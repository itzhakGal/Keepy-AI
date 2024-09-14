from transformers import BertTokenizer, BertForSequenceClassification
import torch
from KeepyAI.config import Config


class PositiveSentenceClassifier:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(Config.POSITIVE_MODEL_DIR)
        self.model = BertForSequenceClassification.from_pretrained(Config.POSITIVE_MODEL_DIR)

    def classify_sentence(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1)
        return "positive" if prediction.item() == 1 else "neutral"

    def get_bert_embeddings(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()
