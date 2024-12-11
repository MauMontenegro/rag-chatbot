from fastapi import FastAPI
from pydantic import BaseModel
from rag import initializeSystem,handle_query

app = FastAPI()

print("Initialize Systems")
initializeSystem()
print("System Initialized")

class QueryRequest(BaseModel):
    query: str

@app.post("/")
def chat(request: QueryRequest):
    query = request.query
    response = handle_query(query)
    return {"response":response}