from openai import OpenAI
from trulens.apps.custom import instrument
from trulens.core import TruSession
from inspeq.client import InspeqEval



session = TruSession()
# session.reset_database()

class RAG:
    @instrument
    def retrieve(self, query: str, vector_store, n_results) -> list:
        """
        Retrieve relevant text from vector store.
        """
        results = vector_store.query(query_texts=query, n_results=n_results)
        # Flatten the list of lists into a single list
        return [doc for sublist in results["documents"] for doc in sublist]

    @instrument
    def generate_completion(self, query: str, context_str: list, messages: list) -> str:
        """
        Generate answer from context.
        """
        oai_client = OpenAI()
        first_query = {
                            "role": "user",
                            "content": f"We have provided context information below. \n"
                            f"---------------------\n"
                            f"{context_str}"
                            f"\n---------------------\n"
                            f"First, say hello and that you're happy to help. \n"
                            f"\n---------------------\n"
                            f"Then, given this information, please answer the question: {query}",
                        }
        if len(messages) > 25:
            return "Maximum context length reached. refresh the page"
        if len(context_str) == 0:
            completion = (
            oai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": f"Please answer the following question: {query}"
                    }
                ],
            )
            .choices[0]
            .message.content
        )
        else: 
            completion = (
                oai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    temperature=0,
                    messages=messages + [first_query]
                )
                .choices[0]
                .message.content
            )
        if completion:
            return completion
        else:
            return "Did not find an answer."

    @instrument
    def query(self, query: str, vector_store, n_results, messages) -> str:
        context_str = self.retrieve(query=query, vector_store = vector_store, n_results = n_results)
        completion = self.generate_completion(
            query=query, context_str=context_str, messages = messages
        )
        return completion
    
def inspeq_result(prompt, context):
    INSPEQ_API_KEY = "hnjngrectuqnszvilzofzrekgxycjppb"
    INSPEQ_PROJECT_ID = "cd563657-3afb-486f-9a32-870378b92e75"
    INSPEQ_API_URL = "https://stage-api.inspeq.ai"# Required only for our on-prem customers
    metrics_list = [
        "NSFW_DETECTION",
        ]
    input_data= [
                {
                    "prompt": prompt,
                    "context": "string",
                    "response": context
                }]

    inspeq_eval = InspeqEval(inspeq_api_key=INSPEQ_API_KEY, inspeq_project_id=INSPEQ_PROJECT_ID,inspeq_api_url= INSPEQ_API_URL)
    results = inspeq_eval.evaluate_llm_task(
            metrics_list=metrics_list,
            input_data=input_data,
            task_name="My Task"
        )
    final_result = float(results["results"][0]["evaluation_details"]["actual_value"])
    return 1 - final_result
