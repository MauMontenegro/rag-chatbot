from llama_index.core import SimpleDirectoryReader

# Load data from source
def LoadDocuments(source:str):
    documents= SimpleDirectoryReader(source).load_data(show_progress=True)
    return documents