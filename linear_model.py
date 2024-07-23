import torch
from transformers import BertModel
import torch.nn as nn
import torch.optim as optim

# Define the model
class QAModel(nn.Module):
    def __init__(self, bert_model, classes=3, dropout_prob=0.4):
        super(QAModel, self).__init__()
        self.bert = bert_model
        self.dropout1 = nn.Dropout(dropout_prob)
        self.linear1 = nn.Linear(bert_model.config.hidden_size, 128)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.linear2 = nn.Linear(128, classes)  # Assuming 3 classes

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        cls_output = self.dropout1(cls_output)  # Apply first dropout
        cls_output = self.linear1(cls_output)  # Apply first linear layer
        cls_output = self.dropout2(cls_output)  # Apply second dropout
        logits = self.linear2(cls_output)  # Apply second linear layer
        return logits
