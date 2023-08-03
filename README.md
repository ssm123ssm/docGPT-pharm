# docGPT - Document Intelligence 

Welcome to **docGPT**, an easily implemented document intelligence program written as an abstraction over langchain. Simply put all the unstructured data (eg PDF files) in the `data` folder and start asking questions about them!

Useful for Retrieval and Summarization tasks.

## Overview

Place your files in the `data` folder. Install dependencies and run

```
from docGPT_core import * 
from RVS import *

set_tokens(OPENAI_TOKEN='OPENAI TOKEN HERE',HF_TOKEN="HF TOKEN HERE")
llm  =  Llm(model_type='gpt-3.5-turbo')
openai_embeddings  =  Embedding(model_type='openai')
persona  =  Persona(personality_type='explainer')

vs = VectorStore(embedding_model=openai_embeddings)
chain = Chain(retriever=vs, llm=llm, persona=persona)
```

You can start chatting with your data now.

`chain.qa(inputs={"query": "Your question"})`

## Installation

Clone this repo. Install dependencies langchain, gradio, numpy and sklearn using `pip install -r requirements.txt`

## Preparation

Import docGPT_core and RVS modules.

```
from docGPT_core import * 
from RVS import *
```
You have to define the OpenAI token, the LLM to be used and the embedding model to be used. The default LLM is OpenAI `gpt-3.5-turbo` and default embedding model is `text-embedding-ada-002`. If you are planning on using Huggingface models, HugingFace token also needs to be set.

```
#setting tokens
set_tokens(OPENAI_TOKEN='OPENAI TOKEN HERE',HF_TOKEN="HF TOKEN HERE")

#setting models
llm  =  Llm(model_type='gpt-3.5-turbo')
openai_embeddings  =  Embedding(model_type='openai')

#setting the personality of the model
persona  =  Persona(personality_type='explainer')
```

 ## Vectorstore creation

Place all the files you want to create the vectorstore for, in the `data` folder and run call the VectorStore function. You can pass the chunk size  and the overlaps if necessary. 

```
vs = VectorStore(embedding_model=openai_embeddings)
```

This might take some time depending on the size of the documents. If the data in the `data` folder need OCR, additional packages including Tesseract OCR needs to be installed.  

You can save the created vectorstore in disk to avoid re-creating it with `vs.save('my_vectorstore.pkl')` and reload it later with `load_vectorstore('my_vectorstore.pkl')`

## Chain for prompting

The QA chain can be initiated with one line of code

`chain = Chain(retriever=vs, llm=llm, persona=persona)`, 

The `retriever` used in the chain object creation in above example is the default retriever attached to the vectorstore. If you want to retrieve n-number of documents, not the default 3 documents, use Retriever class.

Eg: `chain = Chain(retriever=Retriever(vs, k=4), llm=llm, persona=persona)`

and can query the vectorstore with 

`chain.qa(inputs={"query": "Your question"})`

## Summarization

RVS module provides functions for summarization.

`summarize()` function returns the summary of the vectorstore as a passage or as key points.

eg:  `summarize(vectorstore=vs, llm=llm, max_tokens=2500, summary=False, keypoints=True)`

Toggle `summary` and `keypoints` booleans to return either kind of summary. The method uses Representative Vector Summarization method explained in [https://arxiv.org/abs/2308.00479v1](https://arxiv.org/abs/2308.00479v1).
Change the `max_tokens` parameter, depending on the number of chunks to be included and the size of each chunk. Suggestion is to start with a value like 2000, and depending on the answer, fine tune.

### Further control

Mentioned above is the highest level API for creating the QA engine. Please refer this document for finer control.
