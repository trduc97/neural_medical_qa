import json
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd

def load_bioasq_pubmedqa(bioasq_kaggle_path = '/kaggle/input/bioasq-training-12b/training12b_new.json', pubmed_kaggle_path='/kaggle/input/pubmed-qa/pubmed_qa_pga_labeled.parquet'):
    # Load the JSON file
    with open(kaggle_path,'r') as f:
        bioasq_data=json.load(f)
    # Extract yes/no questions directly
    bioasq_yesno = [{
            'id':question['id'],
            'question':question['body'],
            'final_decision':question['exact_answer'],
            'long_answer':question['ideal_answer'], 
            'documents':question['documents']
        }
        for question in bioasq_data['questions'] if question['type'] == 'yesno']
    # Convert the list of yes/no questions to a Pandas DataFrame
    bioasq_df = pd.DataFrame(bioasq_yesno)

    # Convert the DataFrame to a Hugging Face Dataset
    bioasq_dataset = Dataset.from_pandas(bioasq_df)
    # Create a DatasetDict with the 'train' split
    bioasq_data=DatasetDict({'train': bioasq_dataset})

    # Read from parquet and translate to a dataset object
    pubmed_df=pd.read_parquet(pubmed_kaggle_path)
    dataset=Dataset.from_pandas(pubmed_df)
    #Setting into similar format as from huggingface
    dataset_dict = DatasetDict({'train': dataset})
    
    # Load the pubmedqa dataset
    #pubmedqa_data=load_dataset("pubmed_qa","pqa_labeled") # unstable connection

    #Encoding decisions 
    def decision_encode(question):
        labels_map = {'no': 0, 'maybe': 1, 'yes': 2}
        question['decision_encoded'] = labels_map[question['final_decision']]
        return question

    pubmedqa_data=pubmedqa_data.map(decision_encode)
    bioasq_data=pubmedqa_data.map(decision_encode)

    return bioasq_data, pubmedqa_data

def train_val_test_split(datasetdict,train_size=0.75, val_test_ratio=0.6,strat_col='decision_encoded'):
    #Convert dataset to pandas DataFrame
    df = pd.DataFrame(datasetdict['train'])
    test_size=(1-train_size)
    # Define the stratification column
    stratify_col=strat_col

    #First, split for the train set
    train_df, val_test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[stratify_col],
        random_state=42)
    #Then, split the remaining into validation and test
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=val_test_ratio,
        stratify=val_test_df[stratify_col],
        random_state=42)
    # Convert DataFrames back to Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    return train_dataset, val_dataset, test_dataset
