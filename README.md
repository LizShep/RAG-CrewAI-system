# RAG-CrewAI-system

1.  Environment Setup
Install required libraries:
streamlit
crewai
langchain
faiss-cpu
sentence-transformers
Set API keys:
Groq API key → for LLM
(Optional) Tavily API key → for web search

2. Initialize the LLM
Use Groq-hosted model:
llama-3.3-70b-versatile
Configure:
API key
base URL
temperature (controls randomness)

3. Upload PDF (Streamlit UI)
Create file uploader:
st.file_uploader(type="pdf")
Save uploaded file temporarily using:
tempfile.NamedTemporaryFile

4. Load PDF into memory
Use:
PyPDFLoader
Convert PDF into text documents

5. Clean the text
Remove:
short lines
irrelevant symbols/emails
Keep only meaningful content

6. Split into chunks
Use:
RecursiveCharacterTextSplitter
Settings:
chunk size ≈ 1000
overlap ≈ 200
Purpose:
makes retrieval more accurate

7. Create embeddings
Use:
sentence-transformers/all-MiniLM-L6-v2
Convert text chunks → vectors

8. Build vector database (FAISS)
Store embeddings in FAISS index
Save it in:
st.session_state.db
This ensures:
PDF stays available during session

9.  Retrieval Tool (RAG core)
Create tool:
pdf_search(query)
Steps:
search FAISS
return top matching chunks
Rule:
ONLY use uploaded PDF content

10.  Create CrewAI Agents
Router Agent
Decides:
PDF vs general reasoning
Retriever Agent
Uses:
pdf_search tool
Must only use document context
Answer Agent
Converts retrieved chunks → final answer

11.  Create Tasks
Routing Task
Input: question
Output: routing decision
Retrieval Task
Input: routed question
Output: relevant document chunks
Answer Task
Input: retrieved context
Output: final answer

12. Build CrewAI system
Combine:
agents
tasks
Run sequential pipeline:
route → retrieve → answer

13.  Streamlit App Flow
Step 1: Upload PDF
User uploads document
System:
loads PDF
cleans text
splits into chunks
stores in FAISS
Step 2: Ask question
User enters query
Step 3: Run CrewAI
System processes:
routing decision
retrieval from vector DB
final answer generation
