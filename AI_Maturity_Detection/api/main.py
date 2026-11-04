import sys
import os
import tempfile
import uuid
import json

from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware

# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from farmaura_pipeline import run_farmaura
from multiview.multiview_predict import predict_multiple_views

INSPECTION_DIR = BASE_DIR / "results" / "inspections"

MAX_FILE_SIZE = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}

# ==========================================
# APPLICATION CONFIGURATION
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    INSPECTION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    yield


app = FastAPI(
    title="FARMAURA AI API",
    description=(
        "AI-powered mango maturity detection "
        "and multi-view inspection system."
    ),
    version="2.0",
    lifespan=lifespan
)

# ==========================================
# CORS CONFIGURATION
# ==========================================

origins = [
    origin.strip()
    for origin in os.getenv(
        "FARMAURA_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"]
)


# ==========================================
# FILE VALIDATION
# ==========================================

async def save_uploaded_image(upload, destination):

    if not upload.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is missing."
        )

    extension = Path(upload.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are supported."
        )

    content = await upload.read(MAX_FILE_SIZE + 1)

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Each image must be smaller than 10 MB."
        )

    destination.write_bytes(content)

    return destination


# ==========================================
# ROOT
# ==========================================

@app.get("/")
def home():

    return {
        "project": "FARMAURA",
        "version": "2.0",
        "message": "AI API is running",
        "status": "active",
        "features": [
            "Single image prediction",
            "Six-view mango inspection",
            "Inspection history"
        ]
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    model_path = BASE_DIR / "models" / "farmaura_best.keras"

    return {
        "api_status": "active",
        "model_file_available": model_path.exists(),
        "inspection_storage_available": INSPECTION_DIR.exists()
    }


# ==========================================
# SINGLE IMAGE PREDICTION
# ==========================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    with tempfile.TemporaryDirectory() as temp_dir:

        extension = Path(file.filename or "").suffix.lower()

        temp_path = Path(temp_dir) / f"image{extension}"

        await save_uploaded_image(
            file,
            temp_path
        )

        try:

            result = run_farmaura(temp_path)

            if result is None:
                raise HTTPException(
                    status_code=422,
                    detail="Image processing failed."
                )

            return result

        except HTTPException:
            raise

        except Exception as error:

            print(f"Single prediction error: {error}")

            raise HTTPException(
                status_code=500,
                detail="Internal image prediction error."
            )


# ==========================================
# SIX-VIEW MULTI-VIEW PREDICTION
# ==========================================

@app.post("/predict/multiview")
async def predict_multiview(
    front: UploadFile = File(...),
    back: UploadFile = File(...),
    left: UploadFile = File(...),
    right: UploadFile = File(...),
    top: UploadFile = File(...),
    bottom: UploadFile = File(...)
):

    uploads = {
        "front": front,
        "back": back,
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom
    }

    with tempfile.TemporaryDirectory() as temp_dir:

        image_folder = Path(temp_dir)

        try:

            for view, upload in uploads.items():

                extension = Path(
                    upload.filename or ""
                ).suffix.lower()

                destination = (
                    image_folder / f"{view}{extension}"
                )

                await save_uploaded_image(
                    upload,
                    destination
                )

            result = predict_multiple_views(
                image_folder
            )

            return result

        except HTTPException:
            raise

        except FileNotFoundError as error:

            raise HTTPException(
                status_code=422,
                detail=str(error)
            )

        except Exception as error:

            print(f"Multi-view prediction error: {error}")

            raise HTTPException(
                status_code=500,
                detail="Internal multi-view prediction error."
            )


# ==========================================
# INSPECTION HISTORY
# ==========================================

@app.get("/inspections/{inspection_id}")
def get_inspection(inspection_id: str):

    try:
        inspection_id = str(uuid.UUID(inspection_id))

    except ValueError:

        raise HTTPException(
            status_code=400,
            detail="Invalid inspection ID."
        )

    record_path = (
        INSPECTION_DIR /
        f"{inspection_id}.json"
    )

    if not record_path.is_file():

        raise HTTPException(
            status_code=404,
            detail="Inspection record not found."
        )

    try:

        with open(record_path, "r") as file:
            result = json.load(file)

        return result

    except (OSError, json.JSONDecodeError):

        raise HTTPException(
            status_code=500,
            detail="Unable to read inspection record."
        )
