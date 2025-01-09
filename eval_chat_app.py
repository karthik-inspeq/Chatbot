

import streamlit as st
import pandas as pd
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import LanceDB
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from eval_metrics import * 
from inspeq.client import InspeqEval
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from guardrail_RAG import RAG
from trulens.apps.custom import TruCustomApp
from trulens.apps.custom import instrument
from trulens.core.guardrails.base import context_filter, block_output, block_input
import numpy as np
from trulens.core import Feedback
from trulens.core import Select
from trulens.providers.openai import OpenAI
from guardrail_RAG import inspeq_result, inspeq_result_filtered, inspeq_result_unfiltered
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Define API keys
if 'api_key' not in st.session_state: st.session_state['api_key'] = ""
if 'INSPEQ_API_KEY' not in st.session_state: st.session_state['INSPEQ_API_KEY'] = ""
if 'INSPEQ_PROJECT_ID' not in st.session_state: st.session_state['INSPEQ_PROJECT_ID'] = ""
if 'user_turn' not in st.session_state: st.session_state['user_turn'] = False
if 'pdf' not in st.session_state: st.session_state['pdf'] = None
if "embed_model" not in st.session_state: st.session_state['embed_model'] = None
if "vector_store" not in st.session_state: st.session_state['vector_store'] = None
if "metric_name" not in st.session_state: st.session_state['metric_name'] = None
if "score" not in st.session_state: st.session_state['score'] = None
if "label" not in st.session_state: st.session_state['label'] = None
if "url" not in st.session_state: st.session_state['url'] = None
if "rag" not in st.session_state: st.session_state['rag'] = None
if "text_chunks" not in st.session_state: st.session_state['text_chunks'] = None
if "inspeq_result_function" not in st.session_state: st.session_state["inspeq_result_function"] = None
if "guardrail" not in st.session_state: st.session_state["guardrail"] = None
if "real_response" not in st.session_state: st.session_state["real_response"] = ""

os.environ["OPENAI_API_KEY"] = st.session_state["api_key"]
os.environ["INSPEQ_API_KEY"] = st.session_state["INSPEQ_API_KEY"]
os.environ["INSPEQ_PROJECT_ID"] = st.session_state["INSPEQ_PROJECT_ID"] 
# os.environ["INSPEQ_API_URL"] = "https://stage-api.inspeq.ai"

chromadb.api.client.SharedSystemClient.clear_system_cache()
st.set_page_config(page_title="Document Genie", layout="wide")

# def inspeq_final(prompt, context):
#     # rag = RAG()
#     # response = rag.query(prompt, st.session_state['vector_store'], st.session_state["top_k"], st.session_state["messages"])
#     # # response = st.session_state.get("real_response", None)
#     # if response is None:
#     #     raise ValueError("real_response is not set in session state.")
#     # prompt = "string"
#     # context = "string"
#     response = "string"
#     score = inspeq_result(prompt, context, response)
#     return score
class FilteredRAG(RAG):
    # if st.session_state["real_response"]:
    test_filter = Feedback(inspeq_result, name = "nsfw").on_input().on(Select.RecordCalls.retrieve.rets[:]).aggregate(np.mean)
    # test_filter = Feedback(inspeq_final, name = "nsfw").on_input_output()
    @instrument
    @context_filter(
        feedback=test_filter,
        threshold=0.5,
        keyword_for_prompt="query",
    )
    def retrieve(self, query: str, vector_store, n_results) -> list:
        """
        Retrieve relevant text from vector store.
        """
        results = vector_store.query(query_texts=query, n_results=n_results)
        if "documents" in results and results["documents"]:
            return [doc for sublist in results["documents"] for doc in sublist]
        else:
            return []

def scraper(url):
    response = requests.get(url)

    # Step 2: Check if the request was successful
    if response.status_code == 200:
        # Step 3: Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")

        # Step 4: Extract specific content (e.g., all paragraph text)
        paragraphs = soup.find_all("p")  # Finds all paragraph tags
        content = [p.get_text() for p in paragraphs]  # Extracts text from each <p> tag
        print(content)
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
    return content
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def build_vector_store(text):
    text = "the policeman told the theif'I will fucking kill you' as he was running away" + text
    embedding_function = OpenAIEmbeddingFunction(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="text-embedding-ada-002"
)
    # Define chunking parameters
    chunk_size = st.session_state["chunk_size"]  # Define desired chunk size
    chunk_overlap = st.session_state["chunk_overlap"] # Define desired chunk overlap

    # Initialize the text splitter with the desired chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Split the context into chunks
    text_chunks = text_splitter.split_text(text)

    # Initialize Chroma client and create/get a collection
    chroma_client = chromadb.Client()
    st.session_state['vector_store'] = chroma_client.get_or_create_collection(name="nsfw", embedding_function=embedding_function)

    # Generate unique IDs for each chunk (e.g., "nsfw_context_0", "nsfw_context_1", ...)
    ids = [f"ids_{i}" for i in range(len(text_chunks))]

    # Add the text chunks to the vector store
    try:
        st.session_state['vector_store'].add(
            documents=text_chunks,
            ids=ids
        )
        print("Documents added to the vector store successfully.")
    except Exception as e:
        print("Error adding documents to the vector store:", e)
    return text_chunks
# NSFW detection
from inspeq.client import InspeqEval
from pprint import pprint

def fetch_context(query, guardrail):
    rag = RAG()
    if st.session_state["vector_store"]:
        rag_query = st.session_state["rag"].query(query, st.session_state['vector_store'], st.session_state["top_k"], st.session_state["messages"])
        return rag_query

def get_inspeq_evaluation(prompt, response, context, metric):
    inspeq_eval = InspeqEval(inspeq_api_key=st.session_state['INSPEQ_API_KEY'], inspeq_project_id= st.session_state['INSPEQ_PROJECT_ID'])
    input_data = [{
    "prompt": prompt,
    "response": response,
    "context": context
        }]
    metrics_list = metric
    try:
        output = inspeq_eval.evaluate_llm_task(
            metrics_list=metrics_list,
            input_data=input_data,
            task_name="capital_question"
        )
        return output
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def guardrails(eval_result):
    metrics  = []
    scores = []
    labels = []
    metric_labels = []
    responses = []
    for i in range(len(eval_result["guards"]["evaluations"]["results"])):
        name = eval_result["guards"]["evaluations"]["results"][i]["evaluation_details"]["metric_name"]
        new_name = name.replace("_EVALUATION", "")
        verdict = eval_result["guards"]["evaluations"]["results"][i]["evaluation_details"]["threshold"]
        if verdict != None:
            metric_label = eval_result["guards"]["evaluations"]["results"][i]["evaluation_details"]["metric_labels"][0]
            score = eval_result["guards"]["evaluations"]["results"][i]["evaluation_details"]["actual_value"]
            scores.append(score)
            metric_labels.append(metric_label)
            label = verdict[0]
            labels.append(label)
            metrics.append(new_name)
    return metrics,scores,  metric_labels, responses

def evaluate_all(query, context_lis, response, metrics_list):

    context = "\n\n".join(context_lis) if len(context_lis) else "no context"
    
    RESULT = {}

    RESULT["guards"] = {
        "evaluations" : get_inspeq_evaluation(query, response, context, metrics_list)
    }
    
    return RESULT

def main():
    list_of_metrics = ["RESPONSE_TONE", "ANSWER_RELEVANCE", "FACTUAL_CONSISTENCY", "CONCEPTUAL_SIMILARITY", "READABILITY", "COHERENCE", "CLARITY", 
                               "DIVERSITY", "CREATIVITY", "NARRATIVE_CONTINUITY", "GRAMMATICAL_CORRECTNESS", "PROMPT_INJECTION", 
                               "DATA_LEAKAGE", "INSECURE_OUTPUT", "INVISIBLE_TEXT", "TOXICITY", "BLEU_SCORE", "COMPRESSION_SCORE", 
                               "COSINE_SIMILARITY_SCORE", "FUZZY_SCORE", "METEOR_SCORE", "ROUGE_SCORE"]
    with st.sidebar:
        st.session_state['api_key'] = st.text_input("OpenAI API Key:", type="password", key="api_key_input")
        st.session_state['INSPEQ_API_KEY'] = st.text_input("Inspeq API Key:", type="password", key="inspeq_api_key")
        st.session_state['INSPEQ_PROJECT_ID'] = st.text_input("Inspeq Project ID:", type="password", key="inspeq_project_id")
        st.session_state['url'] = st.text_input("Enter Website URL", type="default", key="web_url")
        st.session_state['guardrail'] = st.toggle("Inspeq Shield")

        _ =  st.number_input("Top-K Contxets to fetch", min_value=1, max_value=50, value=3, step=1, key="top_k")
        _ = st.number_input("Chunk Length", min_value=8, max_value=4096, value=512, step=8, key="chunk_size")
        _ = st.number_input("Chunk Overlap Length", min_value=4, max_value=2048, value=64, step=1, key="chunk_overlap")

        if st.session_state["url"]:
            if st.session_state["embed_model"] is None:
                st.session_state["embed_model"] = OpenAIEmbeddingFunction(
                                                api_key=st.session_state['INSPEQ_API_KEY'],
                                                model_name="text-embedding-ada-002",
                                            )
            raw_text = scraper(st.session_state['url'])
            web_data = " ".join(raw_text)
            st.write(web_data)
            st.session_state['text_chunks'] = build_vector_store(web_data)
            st.write(st.session_state['text_chunks'])

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Enter your question:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = fetch_context(prompt, st.session_state['guardrail'])
        st.session_state.messages.append({"role": "assistant", "content": response})

        final_response = response
        st.chat_message("assistant").write(final_response)


if __name__ == "__main__":
    main()
