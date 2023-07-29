import numpy as np
import statistics
import logging
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain import PromptTemplate
from sklearn.cluster import KMeans


def summarize(vectorstore, llm, embedding_dim=1536, max_tokens=10000):
    index = vectorstore.store.index
    num_items = len(vectorstore.store.index_to_docstore_id)
    embedding_dim = embedding_dim
    vectors = []

    for i in range(num_items):
        vectors.append(index.reconstruct(i))

    embedding_matrix = np.array(vectors)
    doc_index = (vectorstore.store.docstore.__dict__['_dict'])
    chunk_tokens = []

    for key, value in doc_index.items():
        chunk_tokens.append(llm.model.get_num_tokens(value.page_content))

    mean_chunk_size = statistics.mean(chunk_tokens)
    target = max_tokens

    if target // mean_chunk_size <= len(chunk_tokens):
        num_clusters = (target // mean_chunk_size).__int__()
    else:
        num_clusters = len(chunk_tokens).__int__()

    logging.warning(f"Number of chunks chosen: {num_clusters}")

    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(embedding_matrix)

    closest_indices = []

    for i in range(num_clusters):
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
        closest_index = np.argmin(distances)
        closest_indices.append(closest_index)

    selected_indices = sorted(closest_indices)
    doc_ids = list(map(vectorstore.store.index_to_docstore_id.get, selected_indices))
    contents = list(map(vectorstore.store.docstore.__dict__['_dict'].get, doc_ids))

    map_prompt = """
    You will be given a single passage of a document. This section will be enclosed in triple backticks (```)
    Your goal is to identify what the passage tries to describe and give the general idea tha passage is discussing, using few sentences. 
    Do not focus on specific details and try to understand the general context. Start with This section is mainly obout,
    ```{text}```
    GENERAL IDEA:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    map_chain = load_summarize_chain(llm=llm.model,
                                     chain_type="stuff",
                                     prompt=map_prompt_template)

    res = []

    logging.warning('Mapping the summaries...')
    for i in contents:
        res.append(map_chain({"input_documents": [i]}))

    res_2 = []
    for i in res:
        res_2.append(i['output_text'])

    summary_map = ''.join(['\n\nSummary: ' + s for s in res_2])
    summary_doc = Document(page_content=summary_map)

    summary_prompt = """
    You will be given a set of summaries of passages from a document. 
    Your goal is to generate a paragraph, explaining the overall content of the document, 
    by looking at the summaries provided below within triple backticks.
    
    ```{text}```
    
    OVERALL CONTENT:
    """
    summary_prompt_template = PromptTemplate(template=summary_prompt, input_variables=["text"])
    summary_chain = load_summarize_chain(llm=llm.model,
                                         chain_type="stuff",
                                         prompt=summary_prompt_template)
    final_summary = summary_chain({"input_documents": [summary_doc]})

    return final_summary['output_text']
