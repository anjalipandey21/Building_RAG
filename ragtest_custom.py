from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Set LLM and embedding
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

# Load your own documents (3 articles in my_data/)
documents = SimpleDirectoryReader("./my_data/").load_data()

# Create index in memory (no saving)
index = VectorStoreIndex.from_documents(documents)

# Query the index
query_engine = index.as_query_engine()
response = query_engine.query("What are some real-world uses or benefits of RAG?")
print(response)
