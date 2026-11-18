import streamlit as st
from config import config
from src.data_extraction import DataExtractor
from src.data_processing import DataProcessor
from src.llm_interface import LLMInterface
from src.query_engine import RAGQueryEngine, generate_initial_facts

st.set_page_config(page_title="RAG Application with LlamaIndex & OpenRouter", layout="wide")

st.title("Question-Answering & Icebreaker Bot")
st.caption("Powered by LlamaIndex and OpenRouter (Meta Llama)")

# Sidebar Configuration
st.sidebar.header("Configuration")
openrouter_key = st.sidebar.text_input(
    "OpenRouter API Key",
    value=config.OPENROUTER_API_KEY,
    type="password",
    help="Get your key at https://openrouter.ai/keys"
)

llama_models = [
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.2-3b-instruct",
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-3-8b-instruct:free",
]

selected_model = st.sidebar.selectbox("LLM Model (OpenRouter)", llama_models, index=0)
custom_model = st.sidebar.text_input("Or custom model name", value="")
model_name = custom_model.strip() if custom_model.strip() else selected_model

top_k = st.sidebar.slider("Top K Retrieved Chunks", min_value=1, max_value=10, value=config.SIMILARITY_TOP_K)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "initial_facts" not in st.session_state:
    st.session_state.initial_facts = None

st.sidebar.markdown("---")
st.sidebar.header("LinkedIn Profile Extraction")
linkedin_url = st.sidebar.text_input(
    "LinkedIn Profile URL",
    value="https://www.linkedin.com/in/leonkatsnelson",
    help="Target LinkedIn URL"
)
use_mock = st.sidebar.checkbox(
    "Use Mock Data (Recommended)",
    value=True,
    help="Loads premade sample profile from cloud storage without requiring a paid Proxycurl subscription."
)
proxycurl_key = st.sidebar.text_input(
    "Proxycurl API Key (Optional)",
    type="password",
    help="Only required if 'Use Mock Data' is unchecked."
)

# Extract & Index LinkedIn Profile Button
if st.sidebar.button("Extract & Index LinkedIn Profile", type="primary"):
    if not openrouter_key:
        st.sidebar.error("Please provide an OpenRouter API Key!")
    else:
        with st.spinner("Extracting LinkedIn profile, processing, and building vector index..."):
            try:
                LLMInterface(api_key=openrouter_key, llm_model=model_name)
                extractor = DataExtractor(data_dir=config.DATA_DIR)
                docs = extractor.extract_from_linkedin(
                    linkedin_profile_url=linkedin_url,
                    api_key=proxycurl_key,
                    mock=use_mock,
                    save_to_data_dir=True
                )
                if not docs:
                    st.sidebar.error("Failed to extract LinkedIn profile data.")
                else:
                    processor = DataProcessor(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
                    nodes = processor.process_documents(docs)
                    engine = RAGQueryEngine(storage_dir=config.STORAGE_DIR, similarity_top_k=top_k)
                    index = engine.build_index(nodes=nodes, persist=True)

                    # Generate initial facts
                    facts = generate_initial_facts(index)
                    st.session_state.initial_facts = facts
                    st.sidebar.success(f"Indexed profile for '{docs[0].metadata.get('full_name', 'User')}'!")
            except Exception as e:
                st.sidebar.error(f"Error during extraction/indexing: {e}")

st.sidebar.markdown("---")
# General Indexing Button in Sidebar
if st.sidebar.button("Rebuild Index from Data Folder"):
    if not openrouter_key:
        st.sidebar.error("Please provide an OpenRouter API Key!")
    else:
        with st.spinner("Extracting and indexing documents from data/ folder..."):
            try:
                LLMInterface(api_key=openrouter_key, llm_model=model_name)
                extractor = DataExtractor(data_dir=config.DATA_DIR)
                docs = extractor.extract_from_directory()
                if not docs:
                    st.sidebar.warning("No files found in data/ folder!")
                else:
                    processor = DataProcessor(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
                    nodes = processor.process_documents(docs)
                    engine = RAGQueryEngine(storage_dir=config.STORAGE_DIR, similarity_top_k=top_k)
                    engine.build_index(nodes=nodes, persist=True)
                    st.sidebar.success(f"Successfully indexed {len(docs)} document(s) with {len(nodes)} chunks!")
            except Exception as e:
                st.sidebar.error(f"Error indexing: {e}")

# Display Initial Facts Banner if available
if st.session_state.initial_facts:
    with st.expander("3 Interesting Career/Education Facts (Icebreaker)", expanded=True):
        st.markdown(st.session_state.initial_facts)

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg:
            with st.expander("View Source Nodes"):
                for src in msg["sources"]:
                    st.markdown(f"**Score:** `{src['score']}`")
                    st.text(src["text"])

# User Chat Input
query = st.chat_input("Ask a question about the profile or documents...")
if query:
    if not openrouter_key:
        st.warning("Please provide your OpenRouter API key in the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner(f"Retrieving context & generating answer with {model_name}..."):
                try:
                    LLMInterface(api_key=openrouter_key, llm_model=model_name)
                    rag_engine = RAGQueryEngine(storage_dir=config.STORAGE_DIR, similarity_top_k=top_k)
                    
                    if (config.STORAGE_DIR / "docstore.json").exists():
                        rag_engine.load_index()
                    else:
                        extractor = DataExtractor(data_dir=config.DATA_DIR)
                        docs = extractor.extract_from_directory()
                        if not docs:
                            st.error("No indexed documents found. Please extract a LinkedIn profile or add files to 'data/'.")
                            st.stop()
                        processor = DataProcessor(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
                        nodes = processor.process_documents(docs)
                        rag_engine.build_index(nodes=nodes, persist=True)

                    response = rag_engine.query(query)
                    st.write(str(response))

                    sources = [
                        {"score": f"{node.score:.4f}" if node.score is not None else "N/A", "text": node.node.get_text()}
                        for node in getattr(response, "source_nodes", [])
                    ]

                    if sources:
                        with st.expander("View Source Nodes"):
                            for src in sources:
                                st.markdown(f"**Score:** `{src['score']}`")
                                st.text(src["text"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": str(response),
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
