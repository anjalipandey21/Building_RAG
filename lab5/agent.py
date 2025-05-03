from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from duckduckgo_search import DDGS
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)

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


multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)
search_tool = FunctionTool.from_defaults(fn=search)

fntools = [multiply_tool, add_tool, subtract_tool, search_tool]


agent = ReActAgent.from_tools(fntools, llm=Settings.llm, max_iterations=15, verbose=True)
response = agent.chat("What is Brad Pitt's age plus 22?")
print(response)
