import json
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split

# Reading the datafiles from kaggle pre-specified path
def load_bioasq_pubmedqa(bioasq_kaggle_path = '/kaggle/input/bioasq-training-12b/training12b_new.json', 
                         pubmed_kaggle_path='/kaggle/input/pubmed-qa/pubmed_qa_pga_labeled.parquet'):
    # processing the BioASQ dataset
    #Load the JSON file
    with open(bioasq_kaggle_path,'r') as f:
        bioasq_data=json.load(f)
    # Mapping similar features of BioASQ to PubMedQA for downstream task
    bioasq_yesno = [{
            'id':question['id'],
            'question':question['body'], # This is the question
            'final_decision':question['exact_answer'], # This is the yes/no decision, there is no maybe in BioASQ
            'long_answer':question['ideal_answer'],  # This is the info to be pair with the question to decide final_decision/exact_answer
            'documents':question['documents'] # Other info unused in our task
        }
        # Extract yes/no questions to differentiate from the other questions such as factoid
        for question in bioasq_data['questions'] if question['type'] == 'yesno']
    # convert list of yes/no qa to a Pandas DataFrame
    bioasq_df = pd.DataFrame(bioasq_yesno)
    # Convert the df to a Huggingface's Dataset format
    bioasq_dataset = Dataset.from_pandas(bioasq_df)
    # create a DatasetDict with the 'train' split for similar format with PubMedQA
    bioasq_data=DatasetDict({'train': bioasq_dataset})
    
    # processing PubMedQA data                       
    # Read from parquet and convert to a Huggingface's Dataset
    pubmed_df=pd.read_parquet(pubmed_kaggle_path)
    dataset=Dataset.from_pandas(pubmed_df,preserve_index=False)
    #Setting into similar format as huggingface datasetdict
    pubmedqa_data = DatasetDict({'train': dataset})
    
    #Encoding decisions 
    def decision_encode(question):
        labels_map = {'no': 0, 'maybe': 1, 'yes': 2}
        question['decision_encoded']=labels_map[question['final_decision']]
        return question

    # Apply the same encoding to pubmedqa and bioasq
    pubmedqa_data=pubmedqa_data.map(decision_encode)
    bioasq_data=pubmedqa_data.map(decision_encode)
    return bioasq_data, pubmedqa_data


# Perform train test split and return a huggingface Dataset objects 
def pubmed_train_test_split(datasetdict,train_size=0.70, 
                         strat_col='decision_encoded'):
    #Convert dataset to pandas df
    df = pd.DataFrame(datasetdict['train'])
    test_size=(1-train_size)
    # Define the stratified column
    stratify_col=strat_col

    #Split like normal
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[stratify_col],
        random_state=42)
    # Convert DataFrames back to Dataset
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)
    return train_dataset, test_dataset

# Reading the result dictionary and convert to dataframe for comparison 
def result_convert(result_dict):
    df = pd.DataFrame({
        'Model': result_dict.keys(),
        'Accuracy':[result_dict[model]['test']['accuracy'] for model in result_dict],
        'Precision':[result_dict[model]['test']['precision'] for model in result_dict],
        'Recall':[result_dict[model]['test']['recall'] for model in result_dict],
        'F1 Score':[result_dict[model]['test']['f1'] for model in result_dict]})
    return df
