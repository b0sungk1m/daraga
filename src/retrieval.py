import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "alice-index"

def retrieve_relevant_chunks(query, current_page, top_k=3):
    """
    1. Query Pinecone with metadata filter for `page_number`.
    2. Retrieve only relevant embeddings for vector search.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    query_embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={
            "input_type": "query"
        }
    )

    results = index.query(
        namespace="example-namespace",
        vector=query_embedding[0].values,
        top_k=top_k,
        include_values=False,
        include_metadata=True,
        filter={"page_number": {"$lte": current_page}}
    )

    relevant_chunks = [match.metadata['text'] for match in results.matches]
    return relevant_chunks

if __name__ == "__main__":
    print("Retrieval module ready!")