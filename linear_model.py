import torch
from transformers import BertModel
import torch.nn as nn
import torch.optim as optim

# Define the model
class QAModel(nn.Module):
    def __init__(self, bert_model, classes=3):
        super(QAModel, self).__init__()
        self.bert = bert_model
        self.linear = nn.Linear(bert_model.config.hidden_size, classes)  # Assuming 3 classes

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.linear(cls_output)
        return logits