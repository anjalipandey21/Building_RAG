from chromadb import PersistentClient
from llama_index.core import PromptTemplate, Settings, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

# Setup LLM and embedding model
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

# Load from persistent Chroma DB
chroma_client = PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("mgs636test")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load index
index = VectorStoreIndex.from_vector_store(vector_store)

# Custom Prompt
custom_prompt = PromptTemplate(
    "You're an AI assistant helping a student. Summarize the course using only the following context:\n\n{context_str}\n\nQuestion: {query_str}\nAnswer:"
)

# Query with similarity_top_k and custom prompt
query_engine = index.as_query_engine(
    llm=Settings.llm,
    similarity_top_k=3,
    text_qa_template=custom_prompt
)

response = query_engine.query("Give me a simple explanation of what this course is about.")
print(response)
