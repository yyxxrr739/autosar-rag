"""Streamlit app for AUTOSAR RAG with document retrieval and chat interface."""

from typing import List
import debugpy
import streamlit as st
# from phi.assistant import Assistant
from phi.document import Document
# from phi.document.reader.website import WebsiteReader
from phi.document.reader.pdf import PDFReader
from phi.utils.log import logger
from assistant import get_rag_assistant

def restart_assistant():
    """Restart assistant."""
    st.session_state["rag_assistant"] = None
    st.session_state["rag_assistant_run_id"] = None
    if "url_scrape_key" in st.session_state:
        st.session_state["url_scrape_key"] += 1
    if "file_uploader_key" in st.session_state:
        st.session_state["file_uploader_key"] += 1
    st.rerun()

def check_model_change(llm_model, embeddings_model):
    """Check llm config."""
    # Set assistant_type in session state
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = llm_model
    # Restart the assistant if assistant_type has changed
    elif st.session_state["llm_model"] != llm_model:
        st.session_state["llm_model"] = llm_model
        restart_assistant()

    if "embeddings_model" not in st.session_state:
        st.session_state["embeddings_model"] = embeddings_model
    # Restart the assistant if assistant_type has changed
    elif st.session_state["embeddings_model"] != embeddings_model:
        st.session_state["embeddings_model"] = embeddings_model
        st.session_state["embeddings_model_updated"] = True
        restart_assistant()

def get_or_create_assistant(llm_model, embeddings_model):
    """Get or create assistant."""
    if "rag_assistant" not in st.session_state or st.session_state["rag_assistant"] is None:
        logger.info(f"---*--- Creating {llm_model} Assistant ---*---")
        rag_assistant = get_rag_assistant(llm_model=llm_model, embeddings_model=embeddings_model)
        st.session_state["rag_assistant"] = rag_assistant
    else:
        rag_assistant = st.session_state["rag_assistant"]

    try:
        st.session_state["rag_assistant_run_id"] = rag_assistant.create_run()
    except Exception:
        st.warning("Could not create assistant, is the database running?")
        return None

    return rag_assistant

def load_assistant_chat_history(rag_assistant):
    assistant_chat_history = rag_assistant.memory.get_chat_history()
    if len(assistant_chat_history) > 0:
        logger.debug("Loading chat history")
        st.session_state["messages"] = assistant_chat_history
    else:
        logger.debug("No chat history found")
        st.session_state["messages"] = [{
            "role": "assistant", 
            "content": "Upload a doc and ask me questions..."
        }]

def load_knowledge_base(rag_assistant):
    """Load knowledge base."""
    if not rag_assistant.knowledge_base:
        return

    # Add PDFs to knowledge base
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 100

    uploaded_file = st.sidebar.file_uploader(
        "Add a PDF :page_facing_up:", type="pdf", key=st.session_state["file_uploader_key"]
    )
    if uploaded_file is not None:
        process_uploaded_file(rag_assistant, uploaded_file)

    if rag_assistant.knowledge_base.vector_db:
        if st.sidebar.button("Clear Knowledge Base"):
            rag_assistant.knowledge_base.vector_db.clear()
            st.sidebar.success("Knowledge base cleared")

def process_uploaded_file(rag_assistant, uploaded_file):
    """Process the uploaded PDF file."""
    alert = st.sidebar.info("Processing PDF...", icon="🧠")
    rag_name = uploaded_file.name.split(".")[0]
    if f"{rag_name}_uploaded" not in st.session_state:
        # chunk_size and separators can be set here
        reader = PDFReader(chunk_size=300)
        rag_documents: List[Document] = reader.read(uploaded_file)
        if rag_documents:
            rag_assistant.knowledge_base.load_documents(rag_documents, upsert=True)
        else:
            st.sidebar.error("Could not read PDF")
        st.session_state[f"{rag_name}_uploaded"] = True
    alert.empty()

def initialize_debugger(enable_debug: bool = False):
    """Initialize debugpy for remote debugging if enabled."""
    if enable_debug and "debugpy_initialized" not in st.session_state:
        st.session_state.debugpy_initialized = True
        debugpy.listen(("localhost", 5678))
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()

def set_page_config():
    """Set page config."""
    st.set_page_config(
        page_title="AUTOSAR RAG",
        page_icon=":orange_heart:",
    )
    st.title("Local RAG for retrieving AUTOSAR documents")
    st.markdown("##### :orange_heart: built using [autosar-rag](https://github.com/yyxxrr739/autosar-rag)")

def update_session_content(session_messages, rag_assistant):
    """Update session content."""
    # Prompt for user input using a chat input box from Streamlit
    if prompt := st.chat_input():
        session_messages.append({"role": "user", "content": prompt})

    # Display existing chat messages
    for message in session_messages:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is from a user, generate a new response
    last_message = session_messages[-1]
    if last_message.get("role") == "user":
        question = last_message["content"]
        with st.chat_message("assistant"):
            response = ""
            resp_container = st.empty()
            for delta in rag_assistant.run(question):
                response += delta  # type: ignore
                resp_container.markdown(response)
            session_messages.append({"role": "assistant", "content": response})

def load_assistant_storage(rag_assistant, llm_model, embeddings_model):
    """Load assistant storage."""
    if rag_assistant.storage:
        rag_assistant_run_ids: List[str] = rag_assistant.storage.get_all_run_ids()
    new_rag_assistant_run_id = st.sidebar.selectbox("Run ID", options=rag_assistant_run_ids)
    if st.session_state["rag_assistant_run_id"] != new_rag_assistant_run_id:
        logger.info(f"---*--- Loading {llm_model} run: {new_rag_assistant_run_id} ---*---")
        st.session_state["rag_assistant"] = get_rag_assistant(
            llm_model=llm_model, embeddings_model=embeddings_model, run_id=new_rag_assistant_run_id
        )
        st.rerun()

def main(enable_debug: bool = False) -> None:
    """Main function for the Streamlit app."""
    # Set page config
    set_page_config()

    # Start debugger on localhost:5678，and ensure listen only once
    initialize_debugger(enable_debug)

    # Side bar configuration for llm and embeddings
    llm_model = st.sidebar.selectbox(
        "Select Model", 
        options=["llama3", "phi3", "openhermes", "llama2"]
    )
    embeddings_model = st.sidebar.selectbox(
        "Select Embeddings",
        options=["nomic-embed-text", "llama3", "openhermes", "phi3"],
        help="When you change the embeddings model, the documents will need to be added again.",
    )

    check_model_change(llm_model, embeddings_model)

    # create instant of assistant
    rag_assistant = get_or_create_assistant(llm_model, embeddings_model)

    load_assistant_chat_history(rag_assistant)

    # update session content with user input and assistant response
    session_messages = st.session_state["messages"]
    update_session_content(session_messages, rag_assistant)

    load_knowledge_base(rag_assistant)

    load_assistant_storage(rag_assistant, llm_model, embeddings_model)

    if st.sidebar.button("New Run"):
        restart_assistant()

if __name__ == "__main__":
    main(enable_debug=False)
