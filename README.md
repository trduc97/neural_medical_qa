# BioMedical question answering

This project aims to examine different approaches to applying neural language model to enhance the result of the natural language inference task of yes-no question-asnwering task, applying in the field of biomedical, which generally would require a comparitively more complex contextual understanding and potential multi-hop reasoning to understand given topics.    

We will be examining different approach, fine-tuning a few pre-trained language model to enhance the result for the task, using the BERT base language model as the control.    
Model 1: BERT-base  
Model 2: ColBERT  
Model 3: LinkBERT-base  
Model 4: PubMedBERT/BIOMEDNLP-base  
Model 5: BioLinkBERT  

With the 5 models, we use 2 datasets BioASQ and PubMedQA for the fine-tuning process, with focus on the PubMedQA as it was designed to require inference to decide the the answer instead of having a direct answer given, and we will also be experimenting using a part of the PubMedQA-artificial dataset to improve the resulting model  

The 2 main notebooks related to this project is also currently available on kaggle   
https://www.kaggle.com/code/trungcnguyn/diss-biomedqa-context  
https://www.kaggle.com/code/trungcnguyn/diss-biomedqa-nocontext  
