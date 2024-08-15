from transformers import BertModel
import torch
import torch.nn as nn
import torch.optim as optim

# a fully connected dense layers for classification
class QAModel(nn.Module):
    def __init__(self, model, classes=3, dropout_prob=0.5):
        super(QAModel, self).__init__()
        # Assigning the pretrained model, named as BERT since all our models are BERT variants
        self.bert=model
        self.dropout1=nn.Dropout(dropout_prob) # first drop out layer 
        self.linear1 = nn.Linear(model.config.hidden_size, 128) # we use a default 128 nodes 
        self.dropout2 = nn.Dropout(dropout_prob) # secondary drop out layer
        # number of classes may vary between BioASQ (2 classes) and PubMedQA (3 classes) but we are prioritizing experimenting with PubmEdQA so default classes at 3
        self.linear2 = nn.Linear(128, classes)  

    # function to feed forward the embedding information to make classification
    def forward(self, input_ids, attention_mask):
        # all BERT using the MLM will include the attention mask
        outputs= self.bert(input_ids=input_ids, attention_mask=attention_mask) 
        cls_output=outputs.last_hidden_state[:, 0, :]  # Taking out the embedding 
        cls_output=self.dropout1(cls_output)  # first dropout
        cls_output= self.linear1(cls_output)  # first fully connect linear layer
        cls_output=self.dropout2(cls_output)  # second dropout layer
        logits= self.linear2(cls_output)  # Apply second linear layer
        return logits
# a relatively exhaustive Bi-directional LSTM classifier for comparison 
class BiLSTMmodel(nn.Module):
    def __init__(self, model, classes=3,lstm_hidden_size=256,lstm_layers=1, dropout_prob=0.5):
        super(BiLSTMmodel, self).__init__()
        self.bert=model
        self.dropout1=nn.Dropout(dropout_prob)
        #BiLSTM layer
        self.lstm= nn.LSTM(input_size=model.config.hidden_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=True)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.linear1 = nn.Linear(lstm_hidden_size * 2, 128)  #Bidirectional hidden size is doubled
        self.dropout3 = nn.Dropout(dropout_prob)
        self.linear2 = nn.Linear(128, classes) 
    # same as for the case of the fully connected layer
    def forward(self, input_ids, attention_mask):
        # all BERT using the MLM will include the attention mask
        outputs=self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output=outputs.last_hidden_state  # BERT outputs
        # Pass BERT outputs through BiLSTM
        lstm_output, _ =self.lstm(sequence_output)
        lstm_output=lstm_output[:, 0, :]  # Use the hidden state of the first
        lstm_output=self.dropout2(lstm_output) # dropout 
        lstm_output=self.linear1(lstm_output) # fully connected layer
        lstm_output=self.dropout3(lstm_output) # dropout again
        logits=self.linear2(lstm_output) #outputing for prediction 
        
        return logits
