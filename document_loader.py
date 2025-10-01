"""
Document loader module for loading multiple PDFs and counting tokens.
"""

from langchain_community.document_loaders import PyPDFLoader
import tiktoken
from pathlib import Path


def load_documents(documents_dir="documents"):
    """
    Load all PDF documents from the specified directory.

    Args:
        documents_dir: Directory containing PDF files

    Returns:
        Tuple of (all_text, token_count, document_names)
    """
    docs_path = Path(documents_dir)
    pdf_files = list(docs_path.glob("*.pdf"))

    all_text = ""
    document_names = []

    for pdf_file in sorted(pdf_files):
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()

        doc_text = "\n\n".join([page.page_content for page in pages])
        all_text += f"\n\n=== Document: {pdf_file.name} ===\n\n{doc_text}"
        document_names.append(pdf_file.name)

    # Count tokens using tiktoken
    encoding = tiktoken.encoding_for_model("gpt-4")
    token_count = len(encoding.encode(all_text))

    return all_text, token_count, document_names


def count_tokens(text, model="gpt-4"):
    """
    Count tokens in text for a specific model.

    Args:
        text: Text to count tokens for
        model: Model name to use for encoding

    Returns:
        Number of tokens
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
