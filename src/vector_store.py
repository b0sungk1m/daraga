from pinecone import Pinecone, ServerlessSpec
import os
from .logger import shared_logger
import time
from dotenv import load_dotenv
from .embedding import embed_text_chunks

# Load environment variables from .env
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "alice-index"

def create_vector_store(input_dir="data"):
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create Pinecone index if it doesn't exist
    cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
    region = os.environ.get('PINECONE_REGION') or 'us-east-1'
    mySpec = ServerlessSpec(cloud=cloud, region=region)

    if not pc.has_index(INDEX_NAME):
        shared_logger.info(f"Creating Pinecone index: {INDEX_NAME}")
        pc.create_index(name=INDEX_NAME, dimension=1024, metric="cosine", spec=mySpec)

    # Connect to the Pinecone index
    index = pc.Index(INDEX_NAME)
    # Wait for the index to be ready
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(1)

    embeddings, rawChunks = embed_text_chunks(f"{input_dir}/alice_chunks.json")
    upsert_vectors(index, embeddings, rawChunks)
    print("Pinecone Vector Store created and/or connected!")

def upsert_vectors(index, embeddings, rawChunks):
    # Prepare the records for upsert
    # Each contains an 'id', the embedding 'values', and the original text as 'metadata'
    records = []

    shared_logger.info("Form metadata for upsert")
    for d, e in zip(rawChunks, embeddings):
        records.append({
            "id": str(d['id']),
            "values": e['values'],
            "metadata": {'page_number': d['metadata']['page_number'], 'sentence_id': d['metadata']['sentence_id']}
        })

    # Upsert the records into the index
    index.upsert(
        vectors=records,
        namespace="alice-namespace"
    )
    shared_logger.info("Upsert complete")

    time.sleep(10)  # Wait for the upserted vectors to be indexed

    shared_logger.info(index.describe_index_stats())

if __name__ == "__main__":
    create_vector_store()