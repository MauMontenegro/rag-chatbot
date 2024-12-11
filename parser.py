from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter

def SentenceDocumentSplitter(docs):
    pipeline= IngestionPipeline(transformations=[SentenceSplitter(chunk_size=512,chunk_overlap=100)])
    nodes= pipeline.run(documents=docs)
    print(f"Generated {len(nodes)} nodes.")   
    return nodes