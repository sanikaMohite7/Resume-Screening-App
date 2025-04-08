from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import app  # uses app.py from your project

app_api = FastAPI()

app_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "Resume-Screening-App/uploaded"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app_api.post("/upload")
async def upload_resume(resume: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, resume.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(resume.file, f)

    # Get structured summary + keywords
    resume_summary = app.predict_resume_keywords(file_path)

    return resume_summary
