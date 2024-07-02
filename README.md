# siamese_semantic_similarity

This project aims to examine different approaches to applying neural networks, especially the Siamese architecture, to perform and enhance the result of semantic search. This result can potentially be applied to improve the Retrieval Augmented Generation (RAG) application, allowing for easier LLM adoption. 

RAG is the current industry standard solution to allow for the marriage between Large Language Models (LLMs) general-purpose responses 
and case-specific documentation of private enterprises. 

Generally, LLMs are trained based on diverse and large publically available datasets to allow for flexible linguistic responses. 
But when the topics are information outside of the trained datasets (For example regarding enterprises' internal policies), 
while LLMs can still respond, the response will most likely be hallucinating and unreliable due to the lack of access to info during the training process.  
Using RAG, relevant information can be retrieved and fed to the LLMs to provide context, allowing for informed responses without spending the cost and time fine-tuning or pretraining case-specific LLMs.

We will be examining a few models architecture and compare the results as below 
Model 1: traditional multi-layer Dense deep neural network with GLoVe embedding
Model 2: RNN + GloVe embedding
Model 3: GRU and Bidirectional GRU + GloVe embedding
Model 4: LSTM and Bidirectional LSTM + GloVe embedding
Model 5: LSTM + GloVe embedding + Siamese architecture (Paragraph<>Topic) + measure distance to determine similarity
Model 5.1: Testing with Manthantan/Euclidean/Cosine distance
Model 6: LSTM + BERT + Siamese architecture(Manhattan distance)

The intuition of the Siamese model application is to allow learning the topics and the available paragraph separately, and then comparing the similarity between the pair, between the actual match and irrelevant results to determine a viable model.

The use of GloVe and BERT is to compare and determine a viable representation of the available data. 

![image](https://github.com/trduc97/siamese_semantic_similarity/assets/52210863/97b6ef57-0495-4c16-9e1e-fc91a3118fa9)


Overall, while applying a Siamese architecture does improve the learning process, resulting in uniform lower loss value but the f1 score of the models does not have a significant improvement.  

![image](https://github.com/trduc97/siamese_semantic_similarity/assets/52210863/3e6676f2-a99b-4041-9aae-d6f1938b0620)
