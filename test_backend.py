import pytest
import os

# Test 1: Environment variables are set
def test_env_variables():
    assert os.getenv("GROQ_API_KEY") is not None, "GROQ_API_KEY is missing!"

# Test 2: Imports work correctly
def test_imports():
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from langgraph.graph import StateGraph
    assert True

# Test 3: Calculator add
from langraph_rag_backend import calculator

def test_calculator_add():
    result = calculator.invoke({"first_num": 2, "second_num": 3, "operation": "add"})
    assert result["result"] == 5

# Test 4: Calculator divide by zero
def test_calculator_divide_by_zero():
    result = calculator.invoke({"first_num": 5, "second_num": 0, "operation": "div"})
    assert "error" in result

# Test 5: RAG tool without PDF
from langraph_rag_backend import rag_tool

def test_rag_no_document():
    result = rag_tool.invoke({"query": "test query", "thread_id": "fake_thread"})
    assert "error" in result

# Test 6: Thread helpers
from langraph_rag_backend import thread_has_document, thread_document_metadata

def test_thread_has_no_document():
    assert thread_has_document("nonexistent_thread") == False

def test_thread_metadata_empty():
    result = thread_docume