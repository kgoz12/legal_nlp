# legal_nlp

This project examines whether an LLM solution could be used to answer a user's question regarding unitary executive theory.

# Authoritative Sources

The reference sources used to answer the user's questions are intentionally limited to those that are relavent to the concept of unitary executive theory. They include:

* Federalist Paper #47
* Federalist Paper #70
* Executive Order #12866

Other sources should be added later, but the scope needs to be limited due to local memory limitations in my current hardware setup. Other sources should include the constitution, other federalist papers, prior executive orders, case law, etc.

# Methodology

Retrieval Augmented Generation (RAG). 

This model was used to generate the sentence embeddings. We use these vectors for the similarity search between the passages in the databse and the user question: 

* https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/blob/main/all-MiniLM-L6-v2.F16.gguf

This model was used to generate the answers to the user's question: 

* https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF

Pylance was used to store the reference text and vectors. Used dot used to measure the distance between question vector and candidate context vectors.

# TODO 

* Fine-tune the sentence embedding & question answering models to be better suited to this problem domain
* Gather additional reference sources
* Make improvements to the text splitting on the reference sources
* Make impovements to the UI
