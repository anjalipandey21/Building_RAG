# 🧠 MGS636 Lab 4 – Building a Local RAG Architecture

**Author:** Anjali Pandey  

---

## 📌 Overview

This project demonstrates a full implementation of a **Retrieval-Augmented Generation (RAG)** system using both **no-code** and **code-based** tools. The lab is divided into two main phases:

- ✅ **Phase 1:** GUI-based testing using **Open WebUI + Ollama**
- ✅ **Phase 2:** Programmatic RAG system using **Python, LlamaIndex, and ChromaDB**

---

## 🧪 Phase 1: Open WebUI – GUI-Based RAG Testing

| Test Description | Highlights |
|------------------|------------|
| **1. Query Without Knowledge Base** | Basic LLM query — vague/generic answer |
| **2. Query With Syllabus KB** | Used `#MGS636 Syllabus`, generated specific, document-grounded answer |
| **3. Custom KB (3 RAG articles)** | Built personalized knowledge base for testing |
| **4. Model Switching** | Verified output differences across LLMs (e.g., `llama3`, `gemma`, `mistral`) |

---

## 🧪 Phase 2: Python-Based RAG – Code Implementation

| File | Functionality |
|------|----------------|
| `ragtest.py` | Ephemeral test of syllabus RAG pipeline |
| `ragcreate.py` | Saves syllabus PDF into persistent ChromaDB |
| `ragrun.py` | Queries saved syllabus index |
| `ragcustom.py` | Bonus: custom prompt + `similarity_top_k=3` |
| `ragtest_custom.py` | Ephemeral test using 3 personal RAG articles |
| `ragcreate_custom.py` | Stores those articles into persistent ChromaDB |
| `ragrun_custom.py` | Queries custom article-based DB with advanced config |

---

## 📁 Folder Structure

