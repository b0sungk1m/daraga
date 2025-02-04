import requests
import json
import os
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from .logger import shared_logger

BOOK_URL = "https://www.gutenberg.org/files/11/11-0.txt"
WORDS_PER_PAGE = 300

def clean_book(text):
    start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"
    
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    shared_logger.info(f"Found start and end markers at indices {start_idx} and {end_idx}")
    return text[start_idx+len(start_marker):end_idx].strip() if start_idx != -1 and end_idx != -1 else text.strip()

def process_book(output_dir="data"):
    response = requests.get(BOOK_URL)
    if response.status_code != 200:
        shared_logger.error(f"Failed to fetch book from {BOOK_URL}")
        return
    
    book_text = clean_book(response.text)

    parser = SentenceSplitter()
    documents = [Document(text=book_text)]
    chunks = parser.get_nodes_from_documents(documents)

    shared_logger.info(f"Total number of chunks: {len(chunks)}")
    shared_logger.info(f"First 5 chunks: {chunks[:5]}")
    word_count = 0
    for i, chunk in enumerate(chunks):
        words_in_chunk = len(chunk.text.split())
        word_count += words_in_chunk
        chunk.metadata = {
            "page_number": word_count // WORDS_PER_PAGE,
            "sentence_id": i * 5,
        }
    shared_logger.info(f"Max word count: {word_count} -> {word_count // WORDS_PER_PAGE} pages")

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/alice_chunks.json", "w", encoding="utf-8") as f:
        json.dump([{"id": c.id_, "text": c.text, "metadata": c.metadata} for i, c in enumerate(chunks)], f, indent=4)
    shared_logger.info("Chunks saved")
    return chunks

if __name__ == "__main__":
    process_book()
    print("Book processed and saved!")