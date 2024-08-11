from pathlib import Path
from autosar_loader import AutosarLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore, Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


cookbook_dir = Path(__file__).parent

pdf_path = cookbook_dir.joinpath("data/AUTOSAR/AUTOSAR_CP_SWS_DiagnosticCommunicationManager-54-78.pdf")

llm_model: str = "phi3"
embeddings_model: str = "nomic-embed-text"

pdf_path = cookbook_dir.joinpath("data/AUTOSAR/AUTOSAR_CP_SWS_DiagnosticCommunicationManager-54-78.pdf")

loader = AutosarLoader(str(pdf_path))
rag_docs = loader.load()

# text_splitter = SemanticChunker(
#     OllamaEmbeddings(model=llm_model), breakpoint_threshold_type="standard_deviation"
# )
# text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
#     encoding_name="cl100k_base", chunk_size=100, chunk_overlap=10
# )
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=600,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
doc_snippets = text_splitter.split_documents(rag_docs)

for snipeet in doc_snippets:
    print(snipeet)
    print()


# db = QdrantVectorStore.from_documents(
#     doc_snippets,
#     OllamaEmbeddings(model=llm_model),
#     location=":memory:",  # Local mode with in-memory storage only
#     collection_name="my_documents",
# )
url = "http://localhost:6333"
api_key = "123456"
db = QdrantVectorStore.from_documents(
    documents=doc_snippets,
    embedding=OllamaEmbeddings(model=llm_model),
    url=url,
    prefer_grpc=True,
    api_key=api_key,
    collection_name="my_documents",
)
# db = QdrantVectorStore.from_existing_collection(
#     embedding=OllamaEmbeddings(model=llm_model),
#     collection_name="my_documents",
#     url=url,
#     api_key=api_key,
# )
query = "periodic transmission request"

# db = Chroma.from_documents(documents=doc_snippets, embedding=OllamaEmbeddings(model=llm_model))
docs = db.similarity_search(query, k=4)
print("query:\n")
for doc in docs:
    print(doc)
    print()

