import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

# Load variables from the .env file
load_dotenv(override=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

BASE_URL = "http://127.0.0.1:11438"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the data directory exists
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

class ChatRequest(BaseModel):
    model: str
    system_prompt: str
    user_message: str

@app.post("/api/chat")
async def chat_with_openrouter(request: ChatRequest):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not found in .env file")

    headers = {
        # "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": request.model,
        "stream": False,
        "messages": [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_message}
        ],
        "options": {
            "num_predict": 1000,
            "num_ctx": 32000,
            "temperature": 0
        }
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{BASE_URL}/api/chat",
                headers=headers,
                json=payload,
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
            # reply = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            reply = data.get("message", {}).get("content", "")
            return {"reply": reply}
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# --- NEW FILE MANAGEMENT ENDPOINTS ---

@app.get("/api/files")
async def list_files():
    """List all Excel and CSV files in the data directory."""
    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.xlsx', '.xls', '.csv'))]
    return {"files": files}

@app.get("/api/files/{filename}")
async def get_file(filename: str):
    """Serve a specific file from the data directory."""
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath)

@app.post("/api/save")
async def save_file(file: UploadFile = File(...)):
    """Save an uploaded file back to the data directory."""
    filepath = os.path.join(DATA_DIR, file.filename)
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())
        print("Saved file:", filepath)
    return {"message": "Saved successfully", "filename": file.filename}

# -------------------------------------

@app.get("/")
async def serve_index():
    return FileResponse("index.html")