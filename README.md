🧠 MGS636 Lab 4 – Building a Local RAG Architecture  
Author: Anjali Pandey

📌 Overview  
This project demonstrates a full implementation of a Retrieval-Augmented Generation (RAG) system using both no-code and code-based tools. The lab is divided into two main phases:

✅ Phase 1: GUI-based testing using Open WebUI + Ollama  
✅ Phase 2: Programmatic RAG system using Python, LlamaIndex, and ChromaDB  

---

🧪 Phase 1: Open WebUI – GUI-Based RAG Testing

| Text Description            | Highlights                                                  |
|-----------------------------|-------------------------------------------------------------|
| 1. Query Without Knowledge Base | Basic LLM query — vague/generic answer                  |
| 2. Query With Syllabus KB       | Used #MGS636 Syllabus, generated specific, document-grounded answer |
| 3. Custom KB (3 RAG articles)   | Built personalized knowledge base for testing            |
| 4. Model Switching              | Verified output differences across LLMs (e.g., llama3, gemma, mistral) |

---

🧪 Phase 2: Python-Based RAG – Code Implementation

| File                  | Functionality                                                     |
|------------------------|-------------------------------------------------------------------|
| `ragtest.py`           | Ephemeral test of syllabus RAG pipeline                          |
| `ragcreate.py`         | Saves syllabus PDF into persistent ChromaDB                      |
| `ragrun.py`            | Queries saved syllabus index                                     |
| `ragcustom.py`         | Bonus: custom prompt + similarity_top_k=3                        |
| `ragtest_custom.py`    | Ephemeral test using 3 personal RAG articles                     |
| `ragcreate_custom.py`  | Stores those articles into persistent ChromaDB                   |
| `ragrun_custom.py`     | Queries custom article-based DB with advanced config             |

---

🤖 MGS636 Lab 5 – Building an Agentic AI & Web App.

📌 Overview  
This lab expands upon the RAG system by implementing an **Agentic AI** that can reason step-by-step using tools like math functions and web search. It also introduces a web interface using **Streamlit** to make the agent accessible to end-users.

✅ Built tool-using AI agent with LlamaIndex’s `ReActAgent`  
✅ Integrated real-time web search via DuckDuckGo  
✅ Performed arithmetic using custom Python functions  
✅ Deployed AI via Streamlit for a usable web app  

---

🧪 Phase 1: Building the Agent

| File        | Functionality                                                      |
|-------------|--------------------------------------------------------------------|
| `agent.py`  | CLI-based AI agent that uses function tools and real-time search  |

🔧 Tools Implemented:
- `multiply(a, b)`
- `add(a, b)`
- `subtract(a, b)`
- `search(query)` – using DuckDuckGo search API

🔁 Agent uses iterative “thought → action → observation” logic with ReAct framework

---

🧪 Phase 2: Creating the Web App

| File         | Functionality                                              |
|--------------|------------------------------------------------------------|
| `webapp.py`  | Streamlit UI allowing user to submit queries to the agent  |

🌐 Features:
- Sidebar title + instructions
- Validated user input box
- Displays answer interactively
- Shows warning for empty input

---

🛠️ How to Run

1. **Install Requirements**

```bash
pip install llama-index llama-index-llms-ollama duckduckgo-search streamlit
ollama run llama2  # or llama3/gemma/mistral
python lab5/agent.py
streamlit run lab5/webapp.py

MGS636_RAG_PROJECT/
│
├── chroma_db/                  # Syllabus KB
├── chroma_db_custom/           # Custom 3-article KB
├── lab4/
│   ├── ragcreate.py
│   ├── ragrun.py
│   ├── ragcustom.py
│   ├── ragtest.py
│   ├── ragcreate_custom.py
│   ├── ragrun_custom.py
│   └── ragtest_custom.py
│
├── lab5/
│   ├── agent.py
│   └── webapp.py
│
└── README.md
