import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# Import backend graph + helper functions
from langraph_rag_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)

# =========================== Utilities ===========================

def generate_thread_id():
    # Generate a unique ID for each new chat thread
    return uuid.uuid4()


def reset_chat():
    # Create new thread and reset UI state
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    # Add thread to sidebar list if not already present
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    # Load stored conversation state from LangGraph checkpointer
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ======================= Session Initialization ===================

# Initialize chat history if not present
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Initialize current thread ID
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

# Load existing threads from backend persistence
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

# Track uploaded documents per thread
if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

# Ensure current thread exists in thread list
add_thread(st.session_state["thread_id"])

# Convenience variables
thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]  # Reverse for newest first
selected_thread = None


# ============================ Sidebar ============================

st.sidebar.title("LangGraph PDF Chatbot")

# Display current thread ID
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

# Start new chat
if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# Show indexed document info (if exists)
if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

# PDF upload widget
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])

if uploaded_pdf:
    # Prevent re-processing same file
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        # Show indexing progress
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

# Past conversation selector
st.sidebar.subheader("Past conversations")

if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in threads:
        if st.sidebar.button(str(thread_id), key=f"side-thread-{thread_id}"):
            selected_thread = thread_id


# ============================ Main Layout ========================

st.title("ChainSage: PDF Chatbot")

# Render existing messages in chat UI
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box
user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    # Add user message to UI history
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Configuration passed to LangGraph
    CONFIG = {
        "configurable": {"thread_id": thread_key},  # Required for persistence
        "metadata": {"thread_id": thread_key},      # Extra run metadata
        "run_name": "chat_turn",                    # Useful for LangSmith tracing
    }

    # Assistant response block
    with st.chat_message("assistant"):
        status_holder = {"box": None}

        # Stream only AI text chunks (skip raw tool blobs)
        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Detect tool usage and show UI status
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream AI message content only
                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    content = message_chunk.content

                    # Handle structured content blocks (newer SDKs)
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                yield block["text"]

                    elif isinstance(content, str):
                        yield content

        # Stream response to UI
        ai_message = st.write_stream(ai_only_stream())

        # Mark tool execution complete
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message to UI history
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    # Show document metadata below response
    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

# Divider between chat and thread switching
st.divider()

# Handle switching to another conversation thread
if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            continue  # Skip raw tool outputs in UI
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        temp_messages.append({"role": role, "content": msg.content})

    # Replace UI message history with selected thread history
    st.session_state["message_history"] = temp_messages

    # Ensure document state exists for this thread
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})

    st.rerun()
