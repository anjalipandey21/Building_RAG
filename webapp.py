from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from duckduckgo_search import DDGS
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
import streamlit as st


Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)

st.sidebar.title("Welcome to the AI Agent App")
st.sidebar.write("Enter a question and click Submit to get an intelligent response using tools like math, search, and more!")

def multiply(a: int, b: int) -> int:
    return a * b

def add(a: int, b: int) -> int:
    return a + b

def subtract(a: int, b: int) -> int:
    return a - b

def search(query: str) -> str:
    req = DDGS()
    response = req.text(query, max_results=3)
    context = ""
    for result in response:
        context += result['body']
    return context

def word_count(text: str) -> int:
    """Count the number of words in the input text"""
    return len(text.split())




multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)
search_tool = FunctionTool.from_defaults(fn=search)
word_count_tool = FunctionTool.from_defaults(fn=word_count)

fntools = [multiply_tool, add_tool, subtract_tool, search_tool, word_count_tool]
fntools = [multiply_tool, add_tool, subtract_tool, search_tool]


agent = ReActAgent.from_tools(fntools, llm=Settings.llm, max_iterations=15, verbose=True)
st.title("AI-Powered Q&A Agent")  

user_query = st.text_input("Ask a question:", "")  

if st.button("Submit"):
    if user_query:
        agent = ReActAgent.from_tools(fntools, llm=Settings.llm, max_iterations=15, verbose=True)
        answer = agent.chat(user_query)
        st.write("Answer:", answer)
    else:
        st.warning("Please enter a question.")

