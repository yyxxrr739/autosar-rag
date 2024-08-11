"""Document parser for documents."""
from phi.document.reader.pdf import PDFReader

def pdf_parser(uploaded_file, chunk_size=1024):
    """Parse a PDF file."""
    reader = PDFReader(chunk_size=chunk_size)
    return reader.read(uploaded_file)
