from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from dotenv import load_dotenv

import torch

load_dotenv()

model_path = "mistralai/Mixtral-8x7B-Instruct-v0.1"

template = """Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, 
                     llm=HuggingFaceHub(repo_id=model_path, 
                                        model_kwargs={"temperature":0, 
                                                      "max_length":64}))

question = "Give me the python code for fibonacci series"

print(llm_chain({"question":question}))
