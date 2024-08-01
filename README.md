# BioMedical question answering

This project aims to examine different approaches to applying neural networks to enhance the result of the question-asnwering task in the field of biomedical research, which generally would require a comparitively more complex context space in order to understand any given topic. 

This result could be applied to improve the Retrieval Augmented Generation (RAG) application, allowing for easier LLM adoption. RAG is the current industry standard solution to allow for the marriage between Large Language Models (LLMs) general-purpose responses and case-specific documentation of private enterprises. Generally, LLMs are trained based on diverse and large publically available datasets to allow for flexible linguistic responses. But when the topics are information outside of the trained datasets (For example regarding enterprises' internal policies), 
while LLMs can still respond, the response will most likely be hallucinative and unreliable due to the lack of access to relevant info during the training process.  
Using RAG, relevant information can be retrieved and fed to the LLMs to provide context, allowing for informed responses without spending the cost and time fine-tuning or pretraining case-specific LLMs.
Currently, the general approach for RAG is applying vector similarity calculation, comparing the input query to processed vectors saved in a vector database. This process can be multi-layer, using BM25 as a quick, high throughput solution to shortlist the top 10 potential passages, then applying a more advanced model, like ColBERT, to rerank for better results. But this process will only return the most similar passage/paragraph to the query, not a guaranteed correct answer (Imagine you browse Netflix for a historical documentary about President Lincoln, but the top result is "Abraham Lincoln: Vampire Hunter" (2012), relevant, but not what you are looking for. 

So this model is expected to act as the 3rd layer for the QA process, validating the top result in a pair of question-answer to see if the answer actually satisfied the question. 

We will be examining a few approaches to enhance the result for the question-answering task \\
Model 1: BERT-base  
Model 2: GPTv2  
Model 3: LinkBERT-base  
Model 4: BIOMED-BERT-base  
Model 5: BioLinkBERT  

With the 4 models, we use 2 datasets BioASQ and PubMedQA for the fine-tuning process for the task, and we will also be experimenting using a part of the PubMedQA-artificial dataset to improve the resulting model
