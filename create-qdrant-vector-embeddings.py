import os
import logging
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

def get_embeddings():
    """Initialize Azure OpenAI embeddings."""
    try:
        logger.info("Initializing Azure OpenAI Embeddings.")
        return AzureOpenAIEmbeddings(azure_deployment="embedding")
    except Exception as e:
        logger.error(f"Error initializing embeddings: {e}", exc_info=True)
        raise

def main():
    try:
        # Load document
        file_path = r"path to your document"
        logger.info(f"Loading document from {file_path}")
        loader = TextLoader(file_path=file_path)
        documents = loader.load()

        # Split document into chunks
        logger.info("Splitting document into chunks.")
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        split_documents = text_splitter.split_documents(documents=documents)
        logger.info(f"Document split into {len(split_documents)} chunks.")

        # Initialize Qdrant client
        QDRANT_URL = os.getenv("QDRANT_URL")
        QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

        if not QDRANT_URL or not QDRANT_API_KEY:
            logger.error("Missing QDRANT_URL or QDRANT_API_KEY environment variables.")
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set.")

        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        collection_name = "IMPORTANT-MuppaSociety"

        if not qdrant_client.collection_exists(collection_name=collection_name):
            qdrant_client.create_collection(
                collection_name=collection_name, 
                vectors_config={
                    "size": 1536,
                    "distance": "Cosine"
                }
            )

        logger.info(f"Connecting to Qdrant collection: {collection_name}")
        client = QdrantVectorStore(
            content_payload_key="content",
            client=qdrant_client,
            collection_name=collection_name,
            embedding=get_embeddings()
        )

        # Add documents to Qdrant
        logger.info("Adding documents to Qdrant.")
        client.add_documents(documents=split_documents)
        logger.info("Documents successfully added to Qdrant.")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
