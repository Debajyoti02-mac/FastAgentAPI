import warnings
warnings.filterwarnings('ignore') 

# Logging : > To track out the behavior of my agent 
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)
logger.info("Application started")

from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")
# PDF fetch 
from langchain_community.document_loaders import PyPDFLoader 
try:
    loader  = PyPDFLoader('Why_Language_Models_Hallucinate_Explainer.pdf')
    pages = loader.load()
except Exception as e :
    print(str(e))
    
# Split the given document 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
text_spliter = RecursiveCharacterTextSplitter(chunk_size=800 , chunk_overlap=100)
texts = text_spliter.split_documents(pages)

chunks=[i.page_content for  i in texts]
metadata = [i.metadata for i in texts]

import hashlib 
ids=[hashlib.md5(chunk.encode('utf-8')).hexdigest() for chunk in chunks]

# VectorDB create 
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction 
embedding_function = DefaultEmbeddingFunction()

client = chromadb.PersistentClient(path="./VectorDB")

collection = client.get_or_create_collection(name="System",embedding_function=embedding_function)

if collection.count() == 0:
    collection.upsert(
        documents=chunks,
        ids=ids, 
        metadatas=metadata
    )

# Sync chunks and BM25 with all stored documents in collection
all_stored = collection.get()
if all_stored and all_stored.get('documents'):
    chunks = all_stored['documents']

# Hybrid connection
from rank_bm25 import BM25Okapi 
token_corpus = [i.split() for i in chunks if i]
tokens = BM25Okapi(token_corpus) if token_corpus else BM25Okapi([["empty"]])

# Langgraph State Building 
from langgraph.graph import StateGraph , START 
from typing import Annotated ,TypedDict 
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver 
memory = MemorySaver()

config = {'configurable':{
    'thread_id':'user_1'
}}

class State(TypedDict):
    messages:Annotated[list,add_messages]
    
# LLM connection 
from langchain_groq import ChatGroq 
import os 
from dotenv import load_dotenv 
load_dotenv()
os.getenv('GROQ_API_KEY')
LLM = ChatGroq(model='openai/gpt-oss-120b')

# Hybrid Retrival 
from langchain_core.messages import AIMessage
def Hybrid_Rag(state:State):   
    query = state['messages'][-1].content
    logger.info(f"RAG query: {query}")
    prompt = f"""Write the query focused for semantic and keyword search:\n{query}"""
    try:
        query_rewrite = LLM.invoke(prompt).content
    except Exception:
        query_rewrite = query
    
    dense_chunks = []
    try:
        cnt = collection.count()
        if cnt > 0:
            result = collection.query(
                query_texts=[query_rewrite],
                n_results=min(5, cnt)
            )
            distances = result['distances'][0] if result.get('distances') else []
            documents = result['documents'][0] if result.get('documents') else []
            threshold = 1.6
            dense_chunks = [doc for dis, doc in zip(distances, documents) if dis < threshold]
            if not dense_chunks and documents:
                dense_chunks = documents[:3]
    except Exception as e:
        logger.warning(f"Dense retrieval warning: {e}")

    # Keyword search connection
    get_docs = []
    try:
        search_words = [w for w in (query_rewrite.split() or query.split()) if len(w) > 1]
        if search_words and chunks:
            scores = tokens.get_scores(search_words)
            sorted_index = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
            get_docs = [chunks[i] for i, score in sorted_index[:10] if score > 0]
            if not get_docs and sorted_index:
                get_docs = [chunks[i] for i, score in sorted_index[:3]]
    except Exception as e:
        logger.warning(f"Keyword retrieval warning: {e}")
    
    rrf_tokens = {}
    for rank, doc in enumerate(dense_chunks):
        rrf_tokens[doc] = rrf_tokens.get(doc, 0) + 1 / (rank + 61)
    for rank, doc in enumerate(get_docs):
        rrf_tokens[doc] = rrf_tokens.get(doc, 0) + 1 / (rank + 61)
    
    marge = sorted(rrf_tokens.items(), key=lambda x: x[1], reverse=True)
    top_docs = [i for i, x in marge[:4]]
    logger.info(f"Final RRF documents: {len(top_docs)}")
    
    available_files = os.listdir(SAFE_DIR) if os.path.exists(SAFE_DIR) else []
    files_info = f"\nFiles currently in sandbox (agent_files): {', '.join(available_files)}" if available_files else ""
    
    if not top_docs:
        content = "No specific vector document match found."
    else:
        content = "\n\n".join(top_docs)

    return {"messages": [AIMessage(content=f"""Use this retrieved document context to answer the user's question:

{content}
{files_info}

Note: If the user asks about or wants to read a specific file or PDF in the sandbox, you have the file_handler tool (read operation) to read files directly.

User question:
{query}""")]}

# Tool creation 
from langchain_core.tools import tool 
from langchain_community.tools import DuckDuckGoSearchRun
import requests 
import numexpr 
# calculator 
@tool
def calculator(execution: str):
    """
    Perform arithmetic calculations.
    """
    if len(execution) > 200:
        return "Expression too long."
    logger.info(f"Calculator request: {execution}")
    try:
        result = numexpr.evaluate(execution)
        return str(result)
    except Exception as e:
        return f"Calculator error: {str(e)}"
#weather   
@tool 
def weather(location:str):
    """
    Fetch the user provided location's Weather
    """
    logger.info(f"Weather request: {location}")
    try:
        response = requests.get(
            f"https://wttr.in/{location}?format=3",
        timeout=10
        )
        response.raise_for_status()
        return response.text

    except Exception as e:
        return f"Weather tool failed: {str(e)}"
# websearch 
@tool
def web_search(query: str):
    """
    Search the web when retrieved context is insufficient.
    """
    logger.info(f"Web search request: {query}")
    if len(query) > 500:
        return "Search query too long."
    try:
        search = DuckDuckGoSearchRun()
        result = search.run(query)

        if not result:
            return "Web search returned no results."

        return result

    except Exception as e:
        return f"Web search failed: {str(e)}"
    

# File operations 

import os

SAFE_DIR = "agent_files"
os.makedirs(SAFE_DIR, exist_ok=True)

SAFE_DIR = "agent_files"
os.makedirs(SAFE_DIR, exist_ok=True)

@tool
def file_handler(operation: str, file_path: str = "", content: str = ""):
    """
    Handles file operations in agent_files directory.
    Supported operations:
      - 'create' or 'write': create or overwrite a file with content
      - 'read' or 'view': read text files or extract text from PDF files
      - 'append': append content to an existing file
      - 'delete': delete a file
      - 'list': list all files currently in agent_files
    """
    op = (operation or "").lower().strip()
    
    if op in ("list", "ls", "dir"):
        files = os.listdir(SAFE_DIR) if os.path.exists(SAFE_DIR) else []
        return f"Files in agent_files: {', '.join(files) if files else 'Directory is empty'}"

    filename = os.path.basename(file_path.strip()) if file_path else ""
    if not filename:
        return "Please specify a file_path or filename."
        
    full_path = os.path.join(SAFE_DIR, filename)
    MAX_CONTENT_SIZE = 200_000

    if op in ("create", "write", "new"):
        if len(content.encode('utf-8')) > MAX_CONTENT_SIZE:
            return "File content exceeds size limit."
        with open(full_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(content)
        return f"File '{filename}' created successfully in agent_files ({len(content)} characters)."

    elif op in ("read", "view", "open", "cat"):
        target = full_path
        if not os.path.exists(target):
            # Check root directory fallback
            if os.path.exists(filename):
                target = filename
            else:
                avail = ", ".join(os.listdir(SAFE_DIR)) if os.path.exists(SAFE_DIR) else ""
                return f"File '{filename}' not found. Available files: {avail if avail else 'None'}"

        # If it's a PDF, extract text using PyPDFLoader so the LLM can read user uploaded PDFs!
        if filename.lower().endswith('.pdf'):
            try:
                loader = PyPDFLoader(target)
                pages = loader.load()
                pdf_text = "\n\n".join([f"--- Page {i+1} ---\n{p.page_content}" for i, p in enumerate(pages)])
                if len(pdf_text) > 8000:
                    return f"PDF '{filename}' ({len(pages)} pages, first 8000 chars):\n\n" + pdf_text[:8000]
                return f"PDF '{filename}' ({len(pages)} pages):\n\n" + pdf_text
            except Exception as e:
                return f"Error reading PDF '{filename}': {str(e)}"

        try:
            with open(target, "r", encoding="utf-8", errors="replace") as f:
                data = f.read()
            if len(data) > 10000:
                return data[:10000] + f"\n... [Truncated {len(data)-10000} remaining characters]"
            return data
        except Exception as e:
            return f"Error reading file '{filename}': {str(e)}"

    elif op in ("append", "add"):
        mode = "a" if os.path.exists(full_path) else "w"
        with open(full_path, mode, encoding="utf-8", errors="replace") as f:
            f.write(content)
        return f"Content successfully added to '{filename}'."

    elif op in ("delete", "remove", "rm"):
        if os.path.exists(full_path):
            os.remove(full_path)
            return f"File '{filename}' deleted."
        return f"File '{filename}' not found."

    else:
        return f"Unknown operation '{operation}'. Supported operations: create, read, append, delete, list."
    
#tool blind 
tools=[weather,calculator,web_search,file_handler]
LLM_tool=LLM.bind_tools(tools)

# connection 
def tool_connection(state: State):
    logger.info("LLM processing request")

    response = LLM_tool.invoke(state["messages"])

    if response.tool_calls:
        logger.info(
            f"Tool selected: {[tool['name'] for tool in response.tool_calls]}"
        )
    else:
        logger.info("No tool selected")

    return {"messages": response}

# connection state 
Graph = StateGraph(State)
from langgraph.prebuilt import tools_condition , ToolNode 

# Edge builder
Graph.add_node('retrival',Hybrid_Rag)
Graph.add_node('LLM_tools',tool_connection)
Graph.add_node('tools',ToolNode(tools))

# Building node 
Graph.add_edge(START,'retrival')
Graph.add_edge('retrival','LLM_tools')
Graph.add_conditional_edges(
    'LLM_tools',
    tools_condition ,
    {"tools": "tools", "__end__": "__end__"},   
)
Graph.add_edge('tools','LLM_tools')

connention = Graph.compile(checkpointer=memory)
connention

from langchain_core.messages import HumanMessage

# API Connection -------------------------------------------------------------------------
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
# create limiter 
limiter = Limiter(key_func=get_remote_address)


from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI ,HTTPException , Depends , Header 
from database import QueryHistory , create_db
app = FastAPI(title="api_system")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
from pydantic import BaseModel ,Field

# Cross origin resource sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Verify Api key (quiet local fallback so UI doesn't require manual auth)
def verify_api_key(x_api_key: str = Header(default=None)):
    if x_api_key is None or x_api_key == "" or x_api_key == "default" or x_api_key == API_KEY:
        return True
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return True

class QusAns(BaseModel):
    question : str = Field(min_length=1,max_length=1000) 
    id : int = Field(default=None , gt=0)
    thread_id: str = Field(default='default',min_length=1,max_length=100)
    limit : int = Field(default=100 , gt=0, le=100)

class SaveFileModel(BaseModel):
    filename: str
    content: str

from fastapi import UploadFile, File

# Upload PDF and update both ChromaDB and BM25 index
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), x_api_key: str = Header(default=None)):
    verify_api_key(x_api_key)

    temp_path = os.path.join(SAFE_DIR, file.filename)
    content_bytes = await file.read()
    with open(temp_path, "wb") as f:
        f.write(content_bytes)

    new_loader = PyPDFLoader(temp_path)
    new_pages = new_loader.load()
    new_texts = text_spliter.split_documents(new_pages)

    new_chunks = [t.page_content for t in new_texts]
    new_metadata = [t.metadata for t in new_texts]
    new_ids = [hashlib.md5(c.encode('utf-8')).hexdigest() for c in new_chunks]

    if new_chunks:
        collection.upsert(documents=new_chunks, ids=new_ids, metadatas=new_metadata)
        global chunks, tokens
        chunks.extend(new_chunks)
        token_corpus = [c.split() for c in chunks if c]
        tokens = BM25Okapi(token_corpus)

    return {"status": "indexed", "filename": file.filename, "chunks_added": len(new_chunks)}

@app.put("/ask")
def update(request:QusAns , db=Depends(create_db) , x_api_key: str = Header(default=None)):
    verify_api_key(x_api_key)
    question = db.query(QueryHistory).filter(QueryHistory.id == request.id).first()
    if not question:
        raise HTTPException(status_code=400 , detail="questions not find out")
    question.question = request.question 
    db.commit()
    db.refresh(question)
    
    return question

@app.get("/ask")
def fetch(db = Depends(create_db),x_api_key: str = Header(default=None)):
    verify_api_key(x_api_key)
    try:
        questions = db.query(QueryHistory).all()
        return questions
    except Exception as e:
        raise HTTPException(status_code=400 , detail=str(e))
    
@app.delete("/ask")
def delete(requests:QusAns , db=Depends(create_db) , x_api_key: str = Header(default=None)):
    verify_api_key(x_api_key)
    question = db.query(QueryHistory).filter(QueryHistory.id == requests.id).first()
    
    if not question:
        raise HTTPException(status_code=400 , detail="no valid question")
    delete = question.id
    db.delete(question)
    db.commit()
    
    return {
        'status':'deleted',
        'delete':delete
    }

'''It basically do a limit option that do basically a limit option that do 5 question per minits'''
@limiter.limit("5/minute")
@app.post("/chat")
def Question_Answer(request:QusAns , db=Depends(create_db) , x_api_key: str = Header(default=None)):
    verify_api_key(x_api_key)

    config = {
        "configurable": {
            "thread_id": request.thread_id
        }
    }
    try:
        result = connention.invoke({
            'messages': [HumanMessage(content=request.question)]
        }, config=config)
        answer = result['messages'][-1].content

        db_question = QueryHistory(
            question = request.question , 
            limit = request.limit , 
            answer = answer
        )

        db.add(db_question)
        db.commit()
        db.refresh(db_question)
        
        return {
            'question': db_question.question,
            'limit': db_question.limit,
            'id': db_question.id , 
            'answer': answer
        }
    except Exception as e :
        err_msg = str(e)
        logger.error(f"Chat execution error: {err_msg}")
        if "rate_limit_exceeded" in err_msg or "429" in err_msg:
            return {
                'question': request.question,
                'limit': request.limit,
                'id': 0,
                'answer': "⚠️ **Groq Rate Limit**: The Groq API rate limit has been reached for this token window. Please wait 2-3 minutes and try again."
            }
        raise HTTPException(status_code=500 , detail=f"Server error: {err_msg}")
    
from fastapi.responses import FileResponse

@app.get("/files/{filename}")
def get_file(filename: str, x_api_key: str = Header(default=None)):
    verify_api_key(x_api_key)
    path = os.path.join(SAFE_DIR, os.path.basename(filename))
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path) 

@app.get("/api/files")
def list_sandbox_files(x_api_key: str = Header(default=None)):
    verify_api_key(x_api_key)
    files = []
    if os.path.exists(SAFE_DIR):
        for f in sorted(os.listdir(SAFE_DIR)):
            fp = os.path.join(SAFE_DIR, f)
            if os.path.isfile(fp):
                files.append({
                    "filename": f,
                    "size": os.path.getsize(fp),
                    "modified": os.path.getmtime(fp)
                })
    return files

# Code Editor API Endpoints
@app.get("/api/file-content/{filename}")
def get_file_content(filename: str, x_api_key: str = Header(default=None)):
    verify_api_key(x_api_key)
    clean_name = os.path.basename(filename)
    path = os.path.join(SAFE_DIR, clean_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return {"filename": clean_name, "content": f.read()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/save-file")
def save_file_endpoint(req: SaveFileModel, x_api_key: str = Header(default=None)):
    verify_api_key(x_api_key)
    clean_name = os.path.basename(req.filename.strip())
    if not clean_name:
        raise HTTPException(status_code=400, detail="Filename required")
    path = os.path.join(SAFE_DIR, clean_name)
    try:
        with open(path, "w", encoding="utf-8", errors="replace") as f:
            f.write(req.content)
        return {"status": "saved", "filename": clean_name, "size": len(req.content)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/delete-file/{filename}")
def delete_file_endpoint(filename: str, x_api_key: str = Header(default=None)):
    verify_api_key(x_api_key)
    clean_name = os.path.basename(filename)
    path = os.path.join(SAFE_DIR, clean_name)
    if os.path.exists(path):
        try:
            os.remove(path)
            return {"status": "deleted", "filename": clean_name}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/api/status")
def get_system_status(x_api_key: str = Header(default=None), db=Depends(create_db)):
    authenticated = bool(x_api_key and x_api_key == API_KEY)
    history_count = 0
    try:
        history_count = db.query(QueryHistory).count()
    except Exception:
        pass
    
    files_in_sandbox = []
    if os.path.exists(SAFE_DIR):
        for f in os.listdir(SAFE_DIR):
            fp = os.path.join(SAFE_DIR, f)
            if os.path.isfile(fp):
                files_in_sandbox.append({
                    "filename": f,
                    "size": os.path.getsize(fp),
                    "modified": os.path.getmtime(fp)
                })
                
    vector_count = 0
    try:
        vector_count = collection.count()
    except Exception:
        pass

    pdf_name = "Why_Language_Models_Hallucinate_Explainer.pdf"
    pdf_exists = os.path.exists(pdf_name)
    pdf_size = os.path.getsize(pdf_name) if pdf_exists else 0

    return {
        "status": "online",
        "authenticated": authenticated,
        "collection_name": "System",
        "vector_chunks": vector_count,
        "query_history_count": history_count,
        "sandbox_files": files_in_sandbox,
        "default_pdf": {
            "filename": pdf_name,
            "exists": pdf_exists,
            "size": pdf_size
        },
        "tools_available": ["calculator", "weather", "web_search", "file_handler"],
        "model": "openai/gpt-oss-120b"
    }

@app.api_route("/pdf/explainer", methods=["GET", "HEAD"])
def get_explainer_pdf():
    pdf_path = "Why_Language_Models_Hallucinate_Explainer.pdf"
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Explainer PDF not found")
    return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_path)

@app.get("/download-db")
def download_db(x_api_key: str = Header(default=None)):
    verify_api_key(x_api_key)
    db_path = ".SQL_DataBase.db"
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Database not found")
    return FileResponse(db_path, filename="database.db")

from fastapi.staticfiles import StaticFiles
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.api_route("/", methods=["GET", "HEAD"])
def serve_ui():
    index_file = os.path.join("static", "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "FastAgentAPI is running. Place index.html in static/"}


        
    
