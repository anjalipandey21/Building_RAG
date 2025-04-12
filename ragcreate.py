from chromadb import PersistentClient
from llama_index.core import SimpleDirectoryReader, Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

# Set up LLM and embedding
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

# Load YOUR OWN documents (3+ custom .txt files)
documents = SimpleDirectoryReader("./my_data/").load_data()

# Create persistent vector DB
chroma_client = PersistentClient(path="./chroma_db_custom")
chroma_collection = chroma_client.get_or_create_collection("mycustomrag")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Index and store
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
