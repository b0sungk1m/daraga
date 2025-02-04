from src.preprocess import process_book
from src.vector_store import create_vector_store
from src.query import generate_answer
from dotenv import load_dotenv
from .logger import shared_logger
import os

def main():
    shared_logger.info("Initializing Book Q&A System...\n")
    
    # Load environment variables from .env
    load_dotenv()

    if os.getenv("CREATE_INDEX") == "TRUE":
        # Step 1: Process book
        process_book()
        # Step 2: Create Vector Store
        create_vector_store()

    print("\nSystem ready! You can now ask questions.\n")

    while True:
        current_page = int(input("Enter current page number: "))
        query = input("Ask a question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        answer = generate_answer(query, current_page)
        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()