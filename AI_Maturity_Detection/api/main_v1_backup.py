import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Locate the project's src directory
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from farmaura_pipeline import run_farmaura

app = FastAPI(
    title="FARMAURA AI API",
    description="Mango maturity detection API",
    version="1.0"
)

# Development configuration.
# Restrict origins before production deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def home():
    return {
        "project": "FARMAURA",
        "message": "AI API is running",
        "status": "active"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    allowed_extensions = {".jpg", ".jpeg", ".png"}

    extension = Path(file.filename or "").suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are supported."
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 10 MB."
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        result = run_farmaura(temp_path)

        if result is None:
            raise HTTPException(
                status_code=422,
                detail="Image processing failed."
            )

        return result

    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
