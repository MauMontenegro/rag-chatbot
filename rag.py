from loader import LoadDocuments
from parser import SentenceDocumentSplitter
from llama_index.core import VectorStoreIndex,StorageContext,load_index_from_storage, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.nomic import NomicEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import Settings
import os
import faiss
from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv

# Settings
load_dotenv()
NOMIC_API = os.getenv("NOMIC_API")
HG_API = os.getenv("HG_API")
DATA_SOURCE="./data"
INDEX_SOURCE="./index"
EMBED_SIZE=768

# Global Variables
query_engine= None

def initializeSystem():
    global query_engine

    # Configure LLM
    Settings.llm= HuggingFaceInferenceAPI(model_name="mistralai/Mistral-7B-v0.1")
    #Settings.llm= Ollama(model="mistral",request_timeout=40)    

    # Configure Embedding
    embed_model = NomicEmbedding(
        api_key=NOMIC_API,
        dimensionality=EMBED_SIZE,
        model_name="nomic-embed-text-v1.5",
    )

    Settings.embed_model=embed_model

    # Initialize index
    faiss_index=faiss.IndexFlatL2(EMBED_SIZE)

    # Load and pre-process Documents
    documents = LoadDocuments(DATA_SOURCE)   
    nodes = SentenceDocumentSplitter(documents)
    
    # Create Index
    if not os.path.exists(INDEX_SOURCE):
        print("Creating Index")
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context=StorageContext.from_defaults(vector_store=vector_store)
        index=VectorStoreIndex(nodes,embed_model=embed_model,storage_context=storage_context,show_progress=True)
        index.storage_context.persist(persist_dir=INDEX_SOURCE)
    else:   
        print("Index Loaded")
        vector_store=FaissVectorStore.from_persist_dir(INDEX_SOURCE)
        storage_context=StorageContext.from_defaults(vector_store=vector_store,persist_dir=INDEX_SOURCE)
        index= load_index_from_storage(storage_context=storage_context)
           
    # Configure Retriever
    retriever = VectorIndexRetriever(index=index,similarity_top_k=2)     

    # Configure Response Synthesizer
    response_synthesizer = get_response_synthesizer()
    
    # Assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,        
    )    

def handle_query(query:str):
    global query_engine
    response = query_engine.query(query)
    return str(response)