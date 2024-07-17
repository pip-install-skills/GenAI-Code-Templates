from typing import Dict
from langchain_community.vectorstores import ElasticsearchStore
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os 
import aiofiles
import uuid

async def save_file_async(file: str, storage_directory: str) -> None:
    # Remove any previous files in the directory
    for filename in os.listdir(storage_directory):
        file_path = os.path.join(storage_directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Save the current file
    file_path = os.path.join(storage_directory, file.filename)

    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

def get_llm(temperature: float = 0.7, max_tokens: int = 8196) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint="URL_HERE",
        openai_api_key="API_KEY",
        openai_api_version="VERSION",
        azure_deployment="DEPLOYMENT_NAME",
        max_tokens=max_tokens,
        temperature = temperature
    )

# Initialize embeddings
def get_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_endpoint='URL_HERE',
        openai_api_key='API_KEY',
        azure_deployment="DEPLOYMENT_NAME",
        openai_api_version="VERSION",
        openai_api_type="PROVIDER_NAME",
    )

class ElasticSearchVectorSearch:
    def __init__(self) -> None:
        load_dotenv()
        self.url = "http://localhost:9200/"

    async def create_vector_search(self, index_name, file: UploadFile):
        contents = await file.read()

        PARENT_DIRECTORY = "docs/"
        FILENAME = file.filename

        os.makedirs(PARENT_DIRECTORY, exist_ok=True)
        await save_file_async(file=file, storage_directory=PARENT_DIRECTORY)

        file_path = os.path.join(PARENT_DIRECTORY, FILENAME)

        with open(file_path, "wb") as f:
            f.write(contents)

        loader = DirectoryLoader("docs", glob=file.filename, loader_cls=PyPDFLoader)
        data = loader.load()
        os.remove(file_path)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(data)
        # Load the embedding model 
        embeddings = get_embeddings()

        ElasticsearchStore.from_documents(
            texts,
            embeddings,
            es_url=self.url,
            index_name=index_name,
        )

        return {
            "message": "Vectors from document created successfully.",
            "index_name": index_name,
            }

    def make_query(self, query: str, index_name: str) -> Dict:
        db = ElasticsearchStore(
            es_url=self.url,
            index_name=index_name,
            embedding=get_embeddings()
        )

        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        qa = RetrievalQA.from_chain_type(
            llm=get_llm(temperature=0),
            chain_type="stuff",
            retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )

        llm_return_object = qa({"query": query})
        json_response = self.build_json_response(llm_return_object)
        return json_response
    
    def build_json_response(self, llm_return_object) -> Dict:
        response = {}
        response['query'] = llm_return_object['query']
        response['result'] = llm_return_object['result']

        source_doc = llm_return_object['source_documents'][0]
        response['page_content'] = source_doc.page_content
        response['source'] = source_doc.metadata.get('source', None)

        return response
    
app = FastAPI()
esVS = ElasticSearchVectorSearch()

@app.post("/upload")
async def create_vector_search(file: UploadFile = File(...)) -> JSONResponse:
    response = await esVS.create_vector_search(file=file, index_name=str(uuid.uuid4()))
    return JSONResponse(content=response, status_code=200)

@app.get("/query")
async def query_txt(query: str = Query(..., title="Query", description="Enter the question"),
                    index_name: str = Query(..., title="Index", description="Enter the index name")):
    response = esVS.make_query(index_name=index_name, query=query)
    return JSONResponse(content=response, status_code=200)
