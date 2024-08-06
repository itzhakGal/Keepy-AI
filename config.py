import os


class Config:
    SERVER_URL = "http://localhost:8080/"
    YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'
    CSV_FILEPATH = 'cry_diagnosis/yamnet_model/yamnet_class_map.csv'
    MODEL_DIR = 'sentence_diagnosis/model'
    CNN_MODEL_PATH = 'cry_diagnosis/machine_learning_model/cnn/model/baby_cry_cnn_model.h5'
    MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
    VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')
    TOKENIZER_PATH = os.path.join(MODEL_DIR, 'tokenizer.pkl')
    BERT_MODEL_PATH = os.path.join(MODEL_DIR, 'bert_model.pkl')
