"""
Text chunking utilities for RAG.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

def chunk_text(text, chunk_size=500, min_length=35):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ".", ",", " ", ""],
    )
    chunks = splitter.split_text(text)
    # Remove invisible unicode chars and filter by min_length
    return [re.sub(r"[\u200b]", "", c) for c in chunks if len(c) > min_length]

def chunk_documents(documents, chunk_size=500, min_length=35):
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc, chunk_size, min_length))
    return all_chunks
