import streamlit as st
import pandas as pd
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

# Define API keys
if 'api_key' not in st.session_state: st.session_state['api_key'] = None
if 'INSPEQ_API_KEY' not in st.session_state: st.session_state['INSPEQ_API_KEY'] = None
if 'INSPEQ_PROJECT_ID' not in st.session_state: st.session_state['INSPEQ_PROJECT_ID'] = None
if 'user_turn' not in st.session_state: st.session_state['user_turn'] = False
if 'pdf' not in st.session_state: st.session_state['pdf'] = None
if "embed_model" not in st.session_state: st.session_state['embed_model'] = None
if "vector_store" not in st.session_state: st.session_state['vector_store'] = None
if "metric_name" not in st.session_state: st.session_state['metric_name'] = None
if "score" not in st.session_state: st.session_state['score'] = None
if "label" not in st.session_state: st.session_state['label'] = None
if "url" not in st.session_state: st.session_state['url'] = None

st.set_page_config(page_title="Document Genie", layout="wide")

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=st.session_state['chunk_size'], chunk_overlap=st.session_state['chunk_overlap'])
    text_chunks = text_splitter.split_text(text)
    st.session_state['vector_store'] = LanceDB.from_texts(text_chunks, st.session_state["embed_model"])

def fetch_context(query):
    return st.session_state['vector_store'].similarity_search(query, k=st.session_state['top_k'])

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
            if label == "Fail" and label != "N/A":
                if new_name == "PROMPT_INJECTION":
                    response = f"{new_name}: It looks like your message contains instructions that could interfere with the chatbot’s normal operation. For security, let's keep questions straightforward. Feel free to ask about any topic, and I'll do my best to help!\n\n"
                    responses.append(response)
                elif new_name == "INVISIBLE_TEXT":
                    response = f"{new_name}: Your message contains hidden text or characters that I couldn't fully interpret. Please rephrase your question clearly, and I'll be happy to assist!\n"
                    responses.append(response)
                elif new_name == "RESPONSE_TONE":
                    response = f"{new_name}: It seems like my response tone might not have been as expected. Let me know if you'd like a different approach, and I'll make sure to adjust!\n"
                    responses.append(response)
                elif new_name == "DATA_LEAKAGE":
                    response = f"{new_name}: For privacy and security, I'm unable to continue with this request as it might involve sensitive or confidential information. Please rephrase or try asking in a different way!\n"
                    responses.append(response)
                elif new_name == "INSECURE_OUTPUT":
                    response = f"{new_name}: I'm unable to provide information in this format as it may pose security risks. Please try rephrasing your question to avoid any sensitive or potentially harmful content.\n"
                    responses.append(response)
                elif new_name == "TOXICITY":
                    response = f"{new_name}: I'm here to help with respectful and constructive conversations. Let’s keep it positive! Please feel free to ask me anything in a considerate way.\n"
                    responses.append(response)
            labels.append(label)
            metrics.append(new_name)
    return metrics,scores,  metric_labels, responses

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, 
    while considering prior context if available. If the answer is not in 
    provided context, say, "I don't think the answer is available in the context".
    
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, openai_api_key=st.session_state['api_key'])
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    # Retrieve the most relevant context
    contexts_with_scores = fetch_context(user_question)
    # Get previous conversation context
    prior_context = "\n".join([msg["content"] for msg in st.session_state["messages"] if msg["role"] == "assistant"])
    
    # Combine prior context with the current relevant document context
    context_combined = prior_context + "\n\n" + "\n".join([doc.page_content for doc in contexts_with_scores])
    
    # Generate a response using the combined context
    chain = get_conversational_chain()
    response = chain({"input_documents": contexts_with_scores, "question": user_question}, return_only_outputs=True)
    
    return contexts_with_scores, response["output_text"]

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

        _ =  st.number_input("Top-K Contxets to fetch", min_value=1, max_value=50, value=3, step=1, key="top_k")
        _ = st.number_input("Chunk Length", min_value=8, max_value=4096, value=512, step=8, key="chunk_size")
        _ = st.number_input("Chunk Overlap Length", min_value=4, max_value=2048, value=64, step=1, key="chunk_overlap")

        # st.session_state["pdf"] = st.file_uploader("Upload your PDF Files...", accept_multiple_files=True, key="pdf_uploader")

        if st.session_state["url"]:
            if st.session_state["embed_model"] is None:
                st.session_state["embed_model"] = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            raw_text = scraper(st.session_state['url'])
            web_data = " ".join(raw_text)
            st.write(web_data)
            build_vector_store(web_data)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Enter your question:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        contexts_with_scores, response = user_input(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})

        eval_result = evaluate_all(prompt, [item.page_content for item in contexts_with_scores], response, list_of_metrics)
        metric_name, score, labels, responses = guardrails(eval_result)
        st.session_state["metric_name"] = metric_name
        st.session_state["score"] = score
        st.session_state["label"] = labels
        guardrail_response = "\n".join(responses)
        if responses:
            final_response = guardrail_response
        else:
            final_response = response
        st.chat_message("assistant").write(final_response)
        with st.expander("Click to see all the evaluation metrics"):

            final_result = {
                "Metric": st.session_state["metric_name"],
                # "Evaluation Result": eval,
                "Score": st.session_state["score"],
                "Label": st.session_state["label"]
            }
            df = pd.DataFrame(final_result)
            st.table(df)

if __name__ == "__main__":
    main()
