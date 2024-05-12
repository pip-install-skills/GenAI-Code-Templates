from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import os

load_dotenv()

chat = AzureChatOpenAI(
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
    max_tokens=5000,
    temperature=1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)
print(chat([HumanMessage(content="Write me a song about sparkling water.")]))
