from llama_index.llms.ollama import Ollama
from src.retrieval import retrieve_relevant_chunks

llm = Ollama(model="hermes3:8b")

def generate_answer(query, current_page):
    relevant_chunks = retrieve_relevant_chunks(query, current_page)
    context = "\n".join(relevant_chunks)

    prompt = f"Answer the question based only on the context provided. If you cannot answer the question with the given context, that is okay. :\n\nContext:\n{context}\n\nQuestion: {query}"
    response = llm.complete(prompt)
    return response

if __name__ == "__main__":
    print("Query module ready!")