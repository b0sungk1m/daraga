from pinecone import Pinecone
import json
from .logger import shared_logger
import os

def embed_text_chunks(chunks_file):
    # Load book chunks
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    shared_logger.info("Loading text chunks to embed")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Embed text chunks
    shared_logger.info("Embedding text chunks...")
    text_chunks = [chunk["text"] for chunk in chunks]
    embeddings = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=text_chunks,
        parameters={"input_type": "passage", "truncate": "END"}
    )
    shared_logger.info("Text embedded created and saved.")
    return embeddings, chunks