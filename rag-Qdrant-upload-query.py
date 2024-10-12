from langchain_community.vectorstores import Qdrant
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredFileLoader
from qdrant_client import QdrantClient

from langchain.memory import ConversationBufferMemory
from cachetools import TTLCache
from fastapi import BackgroundTasks
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import AsyncIteratorCallbackHandler

import os

class BaseClassLoader:
    def __init__(self) -> None:
        load_dotenv()
        self.callback = AsyncIteratorCallbackHandler()

    def get_embeddings(self):
        return AzureOpenAIEmbeddings(
            model=os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT"), 
            openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'), 
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            disallowed_special=()
        )
    
    def get_llm(self, temperature=0.5, max_tokens=4000, seed=None, top_p=None):
        return AzureChatOpenAI(
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            azure_deployment=os.getenv("AZURE_TURBO_DEPLOYMENT"),
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            top_p=top_p
        )
    
    def get_stream_llm(self, temperature=0.5, max_tokens=4000):
        return AzureChatOpenAI(
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            azure_deployment=os.getenv("AZURE_TURBO_DEPLOYMENT"),
            max_tokens=max_tokens,
            temperature=temperature,
            streaming=True,
            verbose=True,
            callbacks=[self.callback],
        )
    
    def split_text(self, texts, chuck_size=10000, chunk_overlap=500):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chuck_size, chunk_overlap=chunk_overlap)
        documents = text_splitter.split_documents(texts)
        return documents

class LLMClassLoader(BaseClassLoader):
    def __init__(self) -> None:
        load_dotenv()
        self.qdrant_memory_cache = TTLCache(maxsize=float('inf'), ttl=1800)
        self.url = os.getenv("QDRANT_URL")
        self.client = QdrantClient(url=self.url, prefer_grpc=False)
        self.compressor = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2")

    def create_vector_db(self, file_path, collection_name):
        loader = UnstructuredFileLoader(file_path)
        documents = loader.load()
        texts = self.split_text(texts=documents)
        # Load the embedding model 
        embeddings = self.get_embeddings()
        # try:
        #     qdrant = Qdrant(
        #         client=client,
        #         collection_name=collection_name,
        #         embeddings=embeddings,
        #     )
        #     qdrant.add_documents(texts, batch_size=20)
        # except UnexpectedResponse:
        #     qdrant = Qdrant.from_documents(
        #         texts,
        #         embeddings,
        #         url=self.url,
        #         prefer_grpc=False,
        #         collection_name=collection_name
        #     )
        self.client.delete_collection(collection_name=collection_name)
        Qdrant.from_documents(
                texts,
                embeddings,
                url=self.url,
                prefer_grpc=False,
                collection_name=collection_name
            )

        return {"message": "Uploaded files successfully"}
    

    def make_query(self, background_tasks: BackgroundTasks, query: str, username: str):
        if username not in self.qdrant_memory_cache:
            self.qdrant_memory_cache[username] = ConversationBufferMemory(k=5, memory_key="chat_history", return_messages=True, output_key="answer")

        collection_name = "faq_files"

        db = Qdrant(client=self.client, embeddings=self.get_embeddings(), collection_name=collection_name)

        prompt_template = """You are an AI assistant tasked with answering questions based on a given context. Your goal is to provide accurate and helpful responses by carefully analyzing the provided information and the user's question.

        First, here is the context you will use to answer questions:

        <context>: {context}
        </context>

        Your task is to answer questions based solely on the information provided in the context above. Do not use any external knowledge or make assumptions beyond what is explicitly stated in the context.

        When a question is presented, follow these steps:

        1. Carefully analyze the question, paying close attention to specific details and nuances. For example, "How do I encash leaves?" and "How do I apply for leave encashment?" are two different questions that may require different answers.

        2. Search the context thoroughly for relevant information. Look for exact matches or closely related concepts to the question at hand.

        3. If you find relevant information, synthesize it into a clear and concise answer. Make sure your response directly addresses the user's question.

        4. If no relevant information is found, state that "Unfortunately, I don’t have the answer to this question right now. Is there another question I can assist with?".

        5. Present your answer in a user-friendly format, using bullet points or numbered lists if appropriate to improve readability.

        6. Your answer must be in the same language as the question asked by the user. If it is Afrikaans, answer should be Afrikaans, if it is English, then it should be in English.

        7. You should focus on grammatical and spelling accuracy when answering in Afrikaans text. This is crucial for maintaining the integrity and clarity of communication in the Afrikaans language. Make sure sentence structure follows Afrikaans syntax rules.

        Remember to always prioritize accuracy and relevance in your responses, sticking strictly to the information provided in the context.

        Now, find the question:

        <question>: {question}
        </question>

        Now generate a helpful answer below:
        """
        PROMPT = PromptTemplate(input_variables=["context", "question"], 
                        template=prompt_template)
        
        retriever = db.as_retriever(search_kwargs={"k": 2})
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor, base_retriever=retriever
        )


        qa = ConversationalRetrievalChain.from_llm(self.get_llm(temperature=0.1, max_tokens=4000, seed=None, top_p=1),
                                           retriever=compression_retriever, 
                                           memory=self.qdrant_memory_cache[username], 
                                           return_source_documents=True,
                                           combine_docs_chain_kwargs={"prompt": PROMPT},
                                          )

        llm_return_object = qa({"question": query})
        return llm_return_object
  

  
