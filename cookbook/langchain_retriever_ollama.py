from pathlib import Path
from langchain import hub
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

cookbook_dir = Path(__file__).parent
# chroma_db_dir = cookbook_dir.joinpath("storage/chroma_db")
embeddings = (
    OllamaEmbeddings(model="llama3")
)
llm = Ollama(model="llama3.1")

# Load, chunk and index the contents of the blog.
state_of_the_union = cookbook_dir.joinpath("data/demo/state_of_the_union.txt")
# -*- Load the document
loader = TextLoader(str(state_of_the_union))
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

output = rag_chain.invoke("How old is Officer Mora?")
print(output)