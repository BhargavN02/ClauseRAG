# src/main.py

from src.query import load_index_and_metadata, retrieve_chunks, generate_answer_with_context

def rag_legal_clause_finder(query: str, top_k: int = 3) -> str:
    """
    High-level function to retrieve relevant contract clauses and
    generate an answer using local embeddings & local LLM.
    """
    index, metadata_list = load_index_and_metadata()
    retrieved = retrieve_chunks(index, metadata_list, query, top_k=top_k)
    answer = generate_answer_with_context(query, retrieved)
    return answer

def main():
    print("Welcome to the Instant Legal Clause Finder (Local Models)!")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("Ask a question about your legal contracts: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        answer = rag_legal_clause_finder(user_input)
        print("\n[Answer]")
        print(answer)
        print("-" * 60 + "\n")

if __name__ == "__main__":
    main()
