# api_wrapper.py

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
# import extract  # your resume extraction logic
import os
import tempfile

app_api = FastAPI()

# Enable CORS for frontend connection
app_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev; restrict to your domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app_api.post("/upload")
async def upload_resume(resume: UploadFile = File(...)):
    contents = await resume.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    response = extract.process_resume(tmp_path)  # extract logic
    os.remove(tmp_path)
    return response
