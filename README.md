# siamese_semantic_similarity

This project aim to examinate different approach to apply neural network, especially the siamese architecture, to perform and enhance the result of semantic search. This result can potentially be apply to improve the application of Retrieval Augmented Generation (RAG), allowing for easier adoption of LLMs. 

RAG is the current industry standard solution to allow for the marriage between Large Language Models (LLMs) general purpose responses 
and case-specific documentations of private enterprises. 

Generally, LLMs are trained base on diverse and large publically available datasets to allow for a flexible linguistic respones. 
But when the topics are information outside of the trained datasets (For example regarding enterprises' internal policies), 
while LLMs can still provide a response, due to the lack of access of info during the training process, the response will most likely be hallucinative and unreliable.  
With the use of RAG, relevant information can be retrieve and feed to the LLMs to provide a context, allowing for informed responses without spending the cost and times fine-tuning or pretraining case-specific LLMs.

