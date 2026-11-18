"""Command-line interface (CLI) for the RAG Application using Llama on OpenRouter."""

import sys
from config import config
from src.data_extraction import DataExtractor
from src.data_processing import DataProcessor
from src.llm_interface import LLMInterface
from src.query_engine import RAGQueryEngine, generate_initial_facts


def run_rag_pipeline():
    print("=" * 60)
    print("RAG Application CLI (Llama on OpenRouter)")
    print("=" * 60)
    print(f"• LLM Model:       {config.LLM_MODEL}")
    print(f"• Embedding Model: {config.EMBEDDING_MODEL}")
    print(f"• Data Directory:  {config.DATA_DIR}")
    print("=" * 60)

    # 1. Initialize LLM & Embeddings
    print("\n[1/4] Initializing LLM and Embedding models...")
    LLMInterface(
        api_key=config.OPENROUTER_API_KEY,
        llm_model=config.LLM_MODEL,
        embedding_model=config.EMBEDDING_MODEL,
        temperature=config.TEMPERATURE
    )

    # 2. Initialize Query Engine Manager
    rag_engine = RAGQueryEngine(
        storage_dir=config.STORAGE_DIR,
        similarity_top_k=config.SIMILARITY_TOP_K
    )

    # 3. Extract and Index
    extractor = DataExtractor(data_dir=config.DATA_DIR)
    
    # Ensure a LinkedIn profile exists in data/ folder or fetch mock
    linkedin_file = config.DATA_DIR / "linkedin_profile.txt"
    if not linkedin_file.exists():
        print("\n[2/4] Fetching LinkedIn profile data (mock mode)...")
        extractor.extract_from_linkedin(mock=True, save_to_data_dir=True)
    else:
        print(f"\n[2/4] Found existing profile at '{linkedin_file.name}'")

    print(f"[3/4] Indexing documents from '{config.DATA_DIR.name}/'...")
    documents = extractor.extract_from_directory()

    if not documents:
        print("❌ Error: No documents found in 'data/' directory.")
        return

    # Process Documents into Chunks/Nodes
    processor = DataProcessor(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    nodes = processor.process_documents(documents)

    # Build & Persist Index
    index = rag_engine.build_index(nodes=nodes, persist=True)

    # 4. Generate 3 Interesting Facts (Icebreaker)
    print("\n[4/4] Generating 3 Interesting Facts (Icebreaker)...")
    try:
        facts = generate_initial_facts(index)
        print("\n" + "-" * 50)
        print("ICEBREAKER FACTS:")
        print("-" * 50)
        print(facts.strip())
        print("-" * 50)
    except Exception as e:
        print(f"Notice: Could not generate initial facts: {e}")

    # 5. Handle One-off argument or start Interactive Chat Loop
    if len(sys.argv) > 1:
        # User passed a question directly via CLI: python main.py "your question here"
        custom_query = " ".join(sys.argv[1:])
        ask_question(rag_engine, custom_query)
        return

    # Interactive Loop
    print("\n You can now ask questions about the profile / documents.")
    print("   Type 'exit' or 'q' to quit.\n")

    while True:
        try:
            user_input = input(" Question: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye! ")
                break

            ask_question(rag_engine, user_input)
            print()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye! ")
            break


def ask_question(rag_engine: RAGQueryEngine, query_str: str):
    """Executes a query and prints the answer along with top source citations."""
    print(f"\nSearching context and querying {config.LLM_MODEL}...")
    try:
        response = rag_engine.query(query_str)
        print("\n" + "=" * 50)
        print(" ANSWER:")
        print("=" * 50)
        print(str(response).strip())
        print("=" * 50)

        # Print sources
        source_nodes = getattr(response, "source_nodes", [])
        if source_nodes:
            print(f"\n📎 Sources ({len(source_nodes)} chunks retrieved):")
            for i, node in enumerate(source_nodes[:2], 1):
                score_str = f"{node.score:.4f}" if node.score is not None else "N/A"
                snippet = node.node.get_text().replace('\n', ' ')[:140]
                print(f"  [{i}] (Score: {score_str}) {snippet}...")
    except Exception as e:
        print(f" Error during query: {e}")


if __name__ == "__main__":
    run_rag_pipeline()
