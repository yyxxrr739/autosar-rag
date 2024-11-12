from typing import Optional
from phi.assistant import Assistant
from phi.llm.ollama import Ollama
from phi.llm.openai import OpenAIChat
from phi.embedder.ollama import OllamaEmbedder
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.vectordb.qdrant import Qdrant
from phi.knowledge.langchain import LangChainKnowledgeBase
from autosar_rag.autosar_loader import AutosarLoader
from autosar_rag.autosar_splitter import AutosarSplitter
from phi.knowledge import AssistantKnowledge

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"


def get_rag_assistant(
    llm_model: str = "llama3",
    embeddings_model: str = "nomic-embed-text",
    instructions: Optional[str] = None,
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Assistant:
    """Get a Local RAG Assistant."""

    # Define the embedder based on the embeddings model
    embedder = OllamaEmbedder(model=embeddings_model, dimensions=4096)
    embeddings_model_clean = embeddings_model.replace("-", "_")

    if embeddings_model == "nomic-embed-text":
        embedder = OllamaEmbedder(model=embeddings_model, dimensions=768)
    elif embeddings_model == "phi3":
        embedder = OllamaEmbedder(model=embeddings_model, dimensions=3072)

    qdrant_url = "http://localhost:6333"
    api_key = "123456"
    collection_name = "autosar_rag_db"
    vector_db = Qdrant(
        collection=collection_name,
        url=qdrant_url,
        api_key=api_key,
        embedder=embedder
    )

    knowledge = AssistantKnowledge(
        vector_db=vector_db,
        # 3 references are added to the prompt
        num_documents=3,
    )

    if llm_model == "gpt-3.5-turbo":
        llm = OpenAIChat(model="gpt-3.5-turbo", stop="</answer>")
    else:
        llm = Ollama(model=llm_model)

    # Default instructions
    default_instructions = [
        "When a user asks a question, you will be provided with information about the question.",
        "Carefully read this information and provide a clear and concise answer to the user.",
        "Do not use phrases like 'based on my knowledge' or 'depending on the information'.",
        "Follow the user's language to answer, if the user's question is in English, you should answer in English, if the user's question is in Chinese, you should answer in Chinese.",
    ]
    # Combine user instructions with default instructions if provided
    if instructions:
        if '\n' in instructions:
            user_instructions = instructions.split('\n')
        else:
            user_instructions = [instructions]
        default_instructions.extend(user_instructions)

    return Assistant(
        name="local_rag_assistant",
        run_id=run_id,
        user_id=user_id,
        llm=llm,
        storage=PgAssistantStorage(table_name="local_rag_assistant", db_url=db_url),
        knowledge_base=knowledge,
        description="You are an AI called 'RAGit' and your task is to answer questions using the provided information",
        instructions=default_instructions,
        # Uncomment this setting adds chat history to the messages
        # add_chat_history_to_messages=True,
        # Uncomment this setting to customize the number of previous messages added from the chat history
        # num_history_messages=3,
        # This setting adds references from the knowledge_base to the user prompt
        add_references_to_prompt=True,
        # This setting tells the LLM to format messages in markdown
        markdown=True,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
    )        
