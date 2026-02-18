from __future__ import annotations


import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
load_dotenv()


import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langgraph-eval-project-new"

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings  
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition




# -------------------
# 1. LLM + embeddings
# -------------------

# Initialize Groq-hosted LLaMA model for fast inference
llm = ChatGroq(model="llama-3.1-8b-instant")

# Initialize HuggingFace embedding model for vector search
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  

# -------------------
# 2. PDF retriever store (per thread)
# -------------------

# Stores FAISS retrievers per chat thread
_THREAD_RETRIEVERS: Dict[str, Any] = {}

# Stores metadata about uploaded documents per thread
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever associated with a specific thread."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Ingest a PDF file:
    1. Save temporarily
    2. Load and split into chunks
    3. Create FAISS vector store
    4. Store retriever for that thread
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    # Save uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        # Load PDF into LangChain document format
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # Split documents into overlapping chunks for better retrieval
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        # Create FAISS vector store from document chunks
        vector_store = FAISS.from_documents(chunks, embeddings)

        # Convert vector store into retriever interface
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        # Store retriever and metadata for this thread
        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

    finally:
        # Clean up temporary file
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------

# Web search tool (fallback when external information is needed)
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Simple arithmetic tool.
    Allows LLM to perform deterministic math instead of reasoning about it.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieval-Augmented Generation (RAG) tool.
    Retrieves relevant chunks from the uploaded PDF for this thread.
    """
    retriever = _get_retriever(thread_id)

    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    # Perform similarity search
    result = retriever.invoke(query)

    # Extract content and metadata from retrieved documents
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


# Register tools with LLM (enables tool calling)
tools = [search_tool, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------

# Define graph state structure (stores conversation messages)
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------

def chat_node(state: ChatState, config=None):
    """
    Core LLM node.
    Decides whether to:
    - Answer directly
    - Call a tool
    """
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    # System instructions guiding tool usage behavior
    system_message = SystemMessage(
    content=(
        "You are a helpful assistant. Answer general knowledge questions directly. "
        "Only use the `rag_tool` if the user is asking about an uploaded PDF document. "
        "Use web search for current information. Use calculator for math."
        f"If rag_tool is needed, use thread_id `{thread_id}`."
    )
)

    # Combine system message with conversation history
    messages = [system_message, *state["messages"]]

    # Invoke LLM with tool-binding enabled
    response = llm_with_tools.invoke(messages, config=config)

    return {"messages": [response]}


# Tool execution node (executes tool calls emitted by LLM)
tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer
# -------------------

# SQLite-based memory persistence
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 7. Graph
# -------------------

# Create LangGraph state machine
graph = StateGraph(ChatState)

# Add nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

# Define graph flow
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

# Compile graph with persistence enabled
chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helpers
# -------------------

def retrieve_all_threads():
    """Return list of all stored thread IDs."""
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    """Check whether a thread has an uploaded document."""
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    """Return metadata for a thread's uploaded document."""
    return _THREAD_METADATA.get(str(thread_id), {})
