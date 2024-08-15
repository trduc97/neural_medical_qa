import os
import numpy as np
import random
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2Model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import gc
import time

# setting seed for reproducibility in the training process
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic= True
    torch.backends.cudnn.benchmark= False

# Using this class to aggregate all training and testing activities  
class Trainandtest:
    def __init__(self, df_train, df_test, stratify_col='decision_encoded', seed=42, context=False):
        self.train_data= df_train # We would want to use the same set of data for when fine-tuning different models for good comparision
        self.test_data= df_test
        # we use a cross entropy loss function
        self.loss_fn = nn.CrossEntropyLoss()
        # the feature/col for when splitting dataset
        self.stratify_col=stratify_col
        # using a GPU whenever there is one for fine-tuning
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # The dictionary to saves the results of different models
        self.results={}
        self.seed=seed
        # When context = True, we will perform reasoning-required setting (full context) instead of reasoning-free setting
        self.context=context
        set_seed(self.seed)

    # Different models will potentially have different tokenizer, so we setup different tokenizer depending on the name
    def initialize_tokenizer(self, model_name, source):
        # This to avoid issues when the input from the dictionary is a tuple instead of a str
        if isinstance(source, tuple):
                source=source[0]
        # GPT is no longer included but it requires an added token
        if 'GPT' in model_name:
            tokenizer=GPT2Tokenizer.from_pretrained(source)
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            return tokenizer
        # LinkBERT and BiolinkBERT are the main reason for this function 
        elif 'BioLinkBERT' in model_name or 'LinkBERT' in model_name:
            return AutoTokenizer.from_pretrained(source)
        # most other pretrained models can use Bert tokenizer
        else:
            return BertTokenizer.from_pretrained(source)

    # converting the text data to input for mmodels
    def encode_data(self, df, tokenizer):
        # determine if the input for the training process going to use full_context (reasoning-required setting) or long_answer (reasoning-free setting) 
        if self.context: 
            pair='full_context'
        else: pair='long_answer'
        inputs=tokenizer(
            text=df['question'], 
            text_pair=df[pair],  # for case of learning relationship between pair of text, huggingface have a predesignate input for the secondary text
            # Adding padding and truncate to maintain a fixed length in the training process
            padding=True,  
            truncation=True, 
            return_tensors='pt',
            # 512 is the predefined length of input in the original BERT
            max_length=128*4
        )
        labels=torch.tensor(df[self.stratify_col])
        return inputs, labels
    # Loading data into batch for training
    def create_dataloader(self, inputs, labels, batch_size):
        dataset=TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # loading in the pre-trained language model base on the input dictionary
    def import_model(self, QAModel, model_name, source, tokenizer):
        # Avoiding wrong form of input
        if isinstance(source, tuple):
            source=source[0]
        # Depending on model, there are different function to load the parameters
        if 'GPT' in model_name:
            model=GPT2Model.from_pretrained(source)
            model.resize_token_embeddings(len(tokenizer))
            model=QAModel(model)
        elif 'BioLinkBERT' in model_name or 'LinkBERT' in model_name:
            model=AutoModel.from_pretrained(source)
            model=QAModel(model)
        else:
            model=BertModel.from_pretrained(source)
            model=QAModel(model)
        return model

    # This is a step to prepare the batches of data, and import the right tokenizer/pretrained models
    def model_compile(self, QAModel, model_name, source, batch_size=64, optimizer='adam', lr=1e-5):
        set_seed(self.seed) # for reproducibility
        batch_size= 16 if 'GPT' in model_name else batch_size
        tokenizer= self.initialize_tokenizer(model_name, source)
        # encoding input
        train_inputs, train_labels= self.encode_data(self.train_data, tokenizer)
        test_inputs, test_labels= self.encode_data(self.test_data, tokenizer)
        # splitting data into batches
        self.train_loader= self.create_dataloader(train_inputs, train_labels, batch_size)
        self.test_loader= self.create_dataloader(test_inputs, test_labels, batch_size)
        # loading the pretrained model
        self.model=self.import_model(QAModel, model_name, source, tokenizer).to(self.device) 
        # determining the optimizer
        if optimizer=='adam':
            self.optimizer=optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer=='adamw': 
            self.optimizer=optim.AdamW(self.model.parameters(), lr=lr)
        elif optimizer=='sgd':
            self.optimizer=optim.SGD(self.model.parameters(), lr=lr)

    # training/fine-tuning process
    def training(self, model_name, epochs=10):
        set_seed(self.seed) # for reproducibility
        # avoiding wrong input format
        if isinstance(model_name, tuple):
            model_name=model_name[0]        
        self.model.train()
        for epoch in range(epochs):
            start_time=time.time()           # Tracking each epoch training time
            total_loss=0                     # tracking loss    
            all_preds=[]
            all_labels=[]
        
            for batch in self.train_loader:
                # the input are feed into the training process, calculate, feed back and optimized 
                b_input_ids, b_attention_mask, b_labels=[t.to(self.device) for t in batch]
                self.optimizer.zero_grad()
                outputs=self.model(b_input_ids, b_attention_mask)
                loss=self.loss_fn(outputs, b_labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                preds=outputs.detach().cpu().numpy()
                label_ids=b_labels.to('cpu').numpy()
                # clearing out to avoid running out of memories 
                del b_input_ids 
                del b_attention_mask 
                del b_labels
                gc.collect()
                torch.cuda.empty_cache()
                all_preds.append(preds)
                all_labels.append(label_ids)
            # calculating result of each epoch
            avg_loss = total_loss / len(self.train_loader)
            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            avg_f1_score = self.calculate_f1_score(all_preds, all_labels)
            end_time = time.time()
            epoch_duration = end_time - start_time
            print(f"Epoch {epoch+1}, Loss: {avg_loss}, F1 Score: {avg_f1_score}, Time: {epoch_duration:.2f} seconds")
        
        self.save_model(model_name)

    # automatically save the trained model into a pre-defined kaggle folder  
    # this folder will be erase once session end and we cannot save into external folder
    def save_model(self, model_name):
        os.makedirs('/kaggle/working/models', exist_ok=True)
        model_path = f'/kaggle/working/models/{model_name}_model.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, model_path)
        print(f"Model saved to {model_path}")
    # Loading back the model (require knowing the correct tokenizer to work properly 
    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {model_path}")
    # Calculate the F1 score
    def calculate_f1_score(self, preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return f1_score(labels_flat, preds_flat, average='weighted')
    # Evaluating the results during training process, including all metric acc/recall/precision/f1 score
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        predictions, true_labels = [], []
        # Running through the batches 
        with torch.no_grad():
            for batch in dataloader:
                b_input_ids, b_attention_mask, b_labels = [t.to(self.device) for t in batch]
                outputs = self.model(b_input_ids, b_attention_mask)
                logits = outputs.detach().cpu().numpy()
                label_ids = b_labels.cpu().numpy()
                predictions.extend(np.argmax(logits, axis=1))
                true_labels.extend(label_ids)
                # clearning out memories to avoid cuda overflow
                del b_input_ids 
                del b_attention_mask 
                del b_labels
                gc.collect()
                torch.cuda.empty_cache()
        # calculate the final results
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _= precision_recall_fscore_support(true_labels, predictions, average='weighted')
    
        return accuracy, precision, recall, f1
        
    # Evaluate the current trained model or loaded trained model and save the final result into the result dictionary
    def val(self, load_model_path=None):
       # in case evaluating an external trained model
        if load_model_path:
            self.load_model(load_model_path)
    
        test_accuracy, test_precision, test_recall, test_f1= self.evaluate(self.test_loader)
        print(f"Test - Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1-Score: {test_f1}")
    
        return {
            'test': {
                'accuracy': test_accuracy,
                'precision': test_precision,
                'recall': test_recall,
                'f1': test_f1
                }
            }
