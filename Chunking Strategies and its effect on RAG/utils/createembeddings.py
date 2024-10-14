import re
from pypdf import PdfReader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)


def clean_text(text):
    # Replace non-breaking spaces with regular spaces
    text = text.replace('\xa0', ' ')
    # Remove multiple spaces, tabs, or newlines
    text = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing spaces
    return text.strip()

def embeddings_creation(file_paths, strategy):
    pdf_texts = []
    for file_path in file_paths:
        reader = PdfReader(file_path)
        pdf_texts.extend([clean_text(p.extract_text()) for p in reader.pages if p.extract_text()])

    # Filter the empty strings
    pdf_texts = [text for text in pdf_texts if text]

    # Combine all PDF text into one string
    combined_text = "\n\n".join(pdf_texts)

    # Define different chunking strategies
    if strategy == "recursive":
        character_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=1000,
            chunk_overlap=0,
        )
        chunks = character_splitter.split_text(combined_text)

    elif strategy == "token":
        token_splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=0, tokens_per_chunk=256
        )
        chunks = token_splitter.split_text(combined_text)

    elif strategy == "fixed_length":
        # Fixed-length chunking (e.g., 1000 characters per chunk)
        chunk_size = 1000
        chunks = [combined_text[i:i + chunk_size] for i in range(0, len(combined_text), chunk_size)]

    elif strategy == "sentence":
        # Sentence-based chunking using basic splitting
        sentences = combined_text.split(". ")
        chunk_size = 10  # Number of sentences per chunk
        chunks = [
            ". ".join(sentences[i:i + chunk_size])
            for i in range(0, len(sentences), chunk_size)
        ]

    elif strategy == "sliding_window":
        # Sliding window chunking with overlap
        chunk_size = 1000
        overlap = 200  # Overlap of 200 characters between chunks
        chunks = [
            combined_text[i:i + chunk_size]
            for i in range(0, len(combined_text), chunk_size - overlap)
        ]
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    # Generate unique IDs for each chunk
    ids = [str(i) for i in range(len(chunks))]

    return ids, chunks
