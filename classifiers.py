import torch
from transformers import BertModel
import torch.nn as nn
import torch.optim as optim

class QAModel(nn.Module):
    def __init__(self, model, classes=3, dropout_prob=0.5):
        super(QAModel, self).__init__()
        self.bert = model
        self.dropout1 = nn.Dropout(dropout_prob)
        self.linear1 = nn.Linear(model.config.hidden_size, 128)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.linear2 = nn.Linear(128, classes)  # number of classes may vary between BioASQ (2 classes) and PubMedQA (3 classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        cls_output = self.dropout1(cls_output)  # Apply first dropout
        cls_output = self.linear1(cls_output)  # Apply first linear layer
        cls_output = self.dropout2(cls_output)  # Apply second dropout
        logits = self.linear2(cls_output)  # Apply second linear layer
        return logits

class BiLSTMmodel(nn.Module):
    def __init__(self, model, classes=3, lstm_hidden_size=256, lstm_layers=1, dropout_prob=0.5):
        super(BiLSTMmodel, self).__init__()
        self.bert = model
        self.dropout1 = nn.Dropout(dropout_prob)
        
        # BiLSTM layer
        self.lstm = nn.LSTM(input_size=model.config.hidden_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=True)
        
        self.dropout2 = nn.Dropout(dropout_prob)
        self.linear1 = nn.Linear(lstm_hidden_size * 2, 128)  # Bidirectional LSTM hidden size is doubled
        self.dropout3 = nn.Dropout(dropout_prob)
        self.linear2 = nn.Linear(128, classes)  # number of classes may vary between BioASQ (2 classes) and PubMedQA (3 classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # BERT outputs
        
        # Pass BERT outputs through BiLSTM
        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = lstm_output[:, 0, :]  # Use the hidden state of the first token (CLS token)
        
        lstm_output = self.dropout2(lstm_output)
        lstm_output = self.linear1(lstm_output)
        lstm_output = self.dropout3(lstm_output)
        logits = self.linear2(lstm_output)
        
        return logits
