import chromadb
from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

# Set up LLM and embedding model
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
Settings.embed_model = embed_model

# Load syllabus PDF from the data folder
documents = SimpleDirectoryReader("./data/").load_data()

# Create ephemeral (in-memory) Chroma DB
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("mgs636test")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build vector index from documents
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)

# Run a test query using our index
query_engine = index.as_query_engine(llm=Settings.llm)
response = query_engine.query("What topics are covered in this class?")
print(response)
