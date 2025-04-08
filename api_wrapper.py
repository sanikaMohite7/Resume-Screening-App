from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend requests (adjust origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        print(f"Received file: {file.filename}")
        # Optional: save file to disk for testing
        with open(f"./uploaded_{file.filename}", "wb") as f:
            f.write(contents)
        return {"filename": file.filename, "status": "success"}
    except Exception as e:
        print("Upload failed:", str(e))
        return {"status": "error", "message": str(e)}
