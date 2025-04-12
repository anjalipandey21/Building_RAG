from chromadb import PersistentClient
from llama_index.core import PromptTemplate, Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

# Set LLM and embeddings
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

# Load persistent DB created earlier
chroma_client = PersistentClient(path="./chroma_db_custom")
chroma_collection = chroma_client.get_or_create_collection("mycustomrag")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load index
index = VectorStoreIndex.from_vector_store(vector_store)

# Customize the prompt
custom_prompt = PromptTemplate(
    "You are an expert assistant helping a user understand Retrieval-Augmented Generation (RAG).\n"
    "Using the provided context, generate a helpful and specific answer.\n\n"
    "Context:\n{context_str}\n\nQuestion:\n{query_str}"
)

# Query the index
query_engine = index.as_query_engine(
    llm=Settings.llm,
    similarity_top_k=3,
    text_qa_template=custom_prompt
)

response = query_engine.query("Summarize the advantages of using RAG with real-world examples.")
print(response)
