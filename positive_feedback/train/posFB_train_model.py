import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import os

os.environ["WANDB_DISABLED"] = "true"

# Load the dataset
data = pd.read_csv('../data.csv', encoding='ISO-8859-1')

# Encode labels
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['sentence'].tolist(), data['label'].tolist(), test_size=0.2,
                                                    random_state=42)

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the data
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=512)


# Create torch dataset
class SentenceDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = SentenceDataset(train_encodings, y_train)
test_dataset = SentenceDataset(test_encodings, y_test)

# Load BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Set up Trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()

# Evaluate the model
predictions, labels, _ = trainer.predict(test_dataset)
preds = np.argmax(predictions, axis=1)
print(classification_report(y_test, preds, target_names=label_encoder.classes_))

# Save the model and tokenizer using save_pretrained (standard method)
model.save_pretrained('positive_model')
tokenizer.save_pretrained('positive_model')

# Additionally, save just the model's weights using torch.save
torch.save(model.state_dict(), 'positive_model/pytorch_model.bin')
