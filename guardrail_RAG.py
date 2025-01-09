from openai import OpenAI
from trulens.apps.custom import instrument
from trulens.core import TruSession
from inspeq.client import InspeqEval
import os

session = TruSession()
session.reset_database()

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
                            # f"\n---------------------\n"
                            # f"First, say hello and that you're happy to help. \n"
                            f"\n---------------------\n"
                            f"Then, given this information, please answer the question: {query}",
                        }
        # if len(messages) > 25:
        #     return "Maximum context length reached. refresh the page"
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
                    messages=[first_query]
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
        session.reset_database()
        context_str = self.retrieve(query=query, vector_store = vector_store, n_results = n_results)
        completion = self.generate_completion(
            query=query, context_str=context_str, messages = messages
        )

        return completion
    
def inspeq_result(prompt,context):
    metrics_list = [
        "TOXICITY", 
        "PROMPT_INJECTION", 
        "INSECURE_OUTPUT", 
        "INVISIBLE_TEXT",
        "DATA_LEAKAGE"

        ]
    input_data= [
                {
                    "prompt": context,
                    "context": context,
                    "response":context
                }
                ]

    inspeq_eval = InspeqEval(inspeq_api_key=os.environ["INSPEQ_API_KEY"], inspeq_project_id= os.environ["INSPEQ_PROJECT_ID"])
    results = inspeq_eval.evaluate_llm_task(
            metrics_list=metrics_list,
            input_data=input_data,
            task_name="filtering"
        )
    # final_result = float(results["results"][i]["evaluation_details"]["actual_value"])
    eval_result = results
    verdict = float(1)
    for i in range(len(eval_result["results"])):
      name = eval_result["results"][i]["evaluation_details"]["metric_name"]
      new_name = name.replace("_EVALUATION", "")
      print(i)
      final_result = eval_result["results"][i]["metric_evaluation_status"]
      print(new_name)
      print(final_result)
      if final_result == 'FAILED' and new_name == "TOXICITY" and final_result != "EVAL_FAIL":
          verdict = float(0)
          break
      elif final_result == 'FAILED' and new_name == "RESPONSE_TONE" and final_result != "EVAL_FAIL":
          verdict = float(0)
          break
      elif final_result == 'FAILED' and new_name == "PROMPT_INJECTION" and final_result != "EVAL_FAIL":
          verdict = float(0)
          break
      elif final_result == 'FAILED' and new_name == "INSECURE_OUTPUT" and final_result != "EVAL_FAIL":
          verdict = float(0)
          break
      elif final_result == 'FAILED' and new_name == "INVISIBLE_TEXT" and final_result != "EVAL_FAIL":
          verdict = float(0)
          break
      else:
          continue
    print(f"the final verdict is {verdict}")
    return verdict

def inspeq_result_unfiltered(prompt,context):
    response = os.environ["response"]
    metrics_list = [
        "TOXICITY", 
        "PROMPT_INJECTION", 
        "INSECURE_OUTPUT", 
        "INVISIBLE_TEXT",
        "DATA_LEAKAGE",
        ]
    prompt = prompt.encode('utf-8').decode('unicode_escape')
    print(prompt)
    input_data= [
                {
                    "prompt": prompt,
                    "context": context,
                    "response":response
                }
                ]

    inspeq_eval = InspeqEval(inspeq_api_key=os.environ["INSPEQ_API_KEY"], inspeq_project_id= os.environ["INSPEQ_PROJECT_ID"])
    results = inspeq_eval.evaluate_llm_task(
            metrics_list=metrics_list,
            input_data=input_data,
            task_name="unfiltered"
        )
    final_result = float(results["results"][0]["evaluation_details"]["actual_value"])
    return final_result

def inspeq_result_filtered(prompt,context):
    response = os.environ["response"]
    metrics_list = [
        "TOXICITY", 
        "PROMPT_INJECTION", 
        "INSECURE_OUTPUT", 
        "INVISIBLE_TEXT",
        "DATA_LEAKAGE"
        ]
    prompt = prompt.encode('utf-8').decode('unicode_escape')
    input_data= [
                {
                    "prompt": prompt,
                    "context": context,
                    "response":response
                }
                ]

    inspeq_eval = InspeqEval(inspeq_api_key=os.environ["INSPEQ_API_KEY"], inspeq_project_id= os.environ["INSPEQ_PROJECT_ID"])
    results = inspeq_eval.evaluate_llm_task(
            metrics_list=metrics_list,
            input_data=input_data,
            task_name="filtered"
        )
    final_result = float(results["results"][0]["evaluation_details"]["actual_value"])
    return final_result


