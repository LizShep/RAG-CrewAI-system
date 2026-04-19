import os
import streamlit as st
import tempfile

from crewai import Crew, Task, Agent, LLM
from crewai.tools import tool

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# =========================
# LLM (GROQ)
# =========================
os.environ["GROQ_API_KEY"] = "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"

llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0.2
)


# =========================
# TEXT PROCESSING
# =========================
def clean_text(text):
    return "\n".join(
        line for line in text.split("\n")
        if len(line.strip()) > 20
    )


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================
# SESSION VECTOR STORE
# =========================
if "db" not in st.session_state:
    st.session_state.db = None


def build_vectorstore(docs):
    chunks = splitter.split_documents(docs)

    clean_chunks = [
        c for c in chunks
        if len(c.page_content.strip()) > 30
    ]

    if st.session_state.db is None:
        st.session_state.db = FAISS.from_documents(clean_chunks, embeddings)
    else:
        st.session_state.db.add_documents(clean_chunks)


# =========================
# TOOL: PDF SEARCH (STRICT)
# =========================
@tool("pdf_search")
def pdf_search(query: str) -> str:
    """
    Search ONLY uploaded PDF content.
    """
    db = st.session_state.db

    if db is None:
        return "No documents uploaded."

    results = db.similarity_search(query, k=4)

    if not results:
        return "No relevant information found in uploaded PDF."

    return "\n\n".join(r.page_content for r in results)


# =========================
# AGENTS (CLEAN + CONTROLLED)
# =========================

router = Agent(
    role="Router",
    goal="Decide if question needs PDF or general reasoning",
    backstory="Routes queries to correct retrieval system",
    llm=llm,
    allow_delegation=False,
    verbose=True
)

retriever = Agent(
    role="Retriever",
    goal="Answer ONLY using PDF context",
    backstory="Must never use outside knowledge",
    tools=[pdf_search],
    llm=llm,
    allow_delegation=False,
    verbose=True
)

answerer = Agent(
    role="Answer Synthesizer",
    goal="Produce final grounded answer",
    backstory="Turns retrieved context into clean answers",
    llm=llm,
    allow_delegation=False,
    verbose=True
)


# =========================
# TASKS
# =========================

route_task = Task(
    description="Decide routing for: {question}",
    expected_output="pdf or general",
    agent=router
)

retrieve_task = Task(
    description="Retrieve relevant PDF context for: {question}",
    expected_output="Relevant chunks only",
    agent=retriever,
    context=[route_task]
)

answer_task = Task(
    description="Answer using ONLY retrieved context",
    expected_output="Final grounded answer",
    agent=answerer,
    context=[retrieve_task]
)


# =========================
# CREW
# =========================
crew = Crew(
    agents=[router, retriever, answerer],
    tasks=[route_task, retrieve_task, answer_task],
    verbose=True
)


# =========================
# STREAMLIT APP
# =========================
st.title("📄 Production RAG + CrewAI")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    docs = PyPDFLoader(path).load()

    for d in docs:
        d.page_content = clean_text(d.page_content)

    build_vectorstore(docs)

    st.success("PDF indexed successfully!")


query = st.text_input("Ask a question")

if st.button("Run"):

    if st.session_state.db is None:
        st.warning("Upload a PDF first")
    else:
        result = crew.kickoff(inputs={"question": query})

        st.subheader("Answer")

        if hasattr(result, "raw"):
            st.write(result.raw)
        else:
            st.write(str(result))