from phi.document.reader.pdf import PDFImageReader
from phi.document import Document
from typing import List
from pathlib import Path
import sys
cookbook_dir = Path(__file__).parent
sys.path.append(str(cookbook_dir.parent))
print(sys.path)
from assistant import get_rag_assistant


llm_model: str = "llama3"
embeddings_model: str = "nomic-embed-text"

# create instant of assistant
rag_assistant = get_rag_assistant(llm_model, embeddings_model)


pdf_path = cookbook_dir.joinpath("data/AUTOSAR/AUTOSAR_CP_SWS_DiagnosticCommunicationManager-54-78.pdf")

separators: List[str] = ["\n", "\n\n", "\r", "\r\n", "\n\r", "\t", " ", "  "]

# chunk_size and separators can be set here
reader = PDFImageReader(chunk_size=300, separators=separators)
rag_documents: List[Document] = reader.read(pdf=pdf_path)

rag_assistant.knowledge_base.load_documents(rag_documents, upsert=True)

doc_retrieved = rag_assistant.knowledge_base.search("authentication")

print(doc_retrieved)

output: str = str()
for doc in rag_documents:
    output += str(doc) + "\n\n"

open("./output.txt", mode="wb+").write(bytes(output, "utf-8"))

