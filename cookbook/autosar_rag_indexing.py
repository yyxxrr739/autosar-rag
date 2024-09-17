from pathlib import Path
from autosar_loader import AutosarLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore, Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter

cookbook_dir = Path(__file__).parent

pdf_path = cookbook_dir.joinpath("data/AUTOSAR/AUTOSAR_CP_SWS_DiagnosticCommunicationManager.pdf")

llm_model: str = "phi3"
embeddings_model: str = "nomic-embed-text"

loader = AutosarLoader(str(pdf_path))
rag_docs = loader.load()

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


url = "http://localhost:6333"
api_key = "123456"
db = QdrantVectorStore.from_documents(
    documents=doc_snippets,
    embedding=OllamaEmbeddings(model=embeddings_model),
    url=url,
    prefer_grpc=True,
    api_key=api_key,
    collection_name="autosar_rag_db",
)

query = "periodic transmission request"


docs = db.similarity_search(query, k=4)

print("query:\n")
for doc in docs:
    print(doc)
    print()

