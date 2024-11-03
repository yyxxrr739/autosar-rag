from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from collections import Iterable
from typing import List


class AutosarSplitter:
    def __init__(self, chunk_size=300, chunk_overlap=0):
        self.text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
     
    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        print(documents)
        doc_snippets = self.text_splitter.split_documents(documents)
        return doc_snippets 
