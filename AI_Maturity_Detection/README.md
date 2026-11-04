# FARMAURA AI – Mango Maturity Detection

## Developer
Hiten G

## Project
FARMAURA 2.0 – Intelligent Fruit Lifecycle and Predictive Harvest Management System

## AI Features
- Mango maturity classification
- Unripe, Partially Ripe and Ripe classes
- MobileNetV3Small model
- Single-image prediction
- Six-view mango prediction
- Image-quality validation
- Majority voting and average-score aggregation
- Manual-review handling
- Unique inspection IDs
- JSON inspection history
- FastAPI integration

## Model
Architecture: MobileNetV3Small

Input: 224 x 224 RGB

Classes:
1. Partially Ripe
2. Ripe
3. Unripe

Dataset test accuracy: 92.92%

Note: This is a dataset-specific result and does not guarantee field accuracy.

## Local Setup

Create a virtual environment:

python3 -m venv tfenv

Activate:

source tfenv/bin/activate

Install dependencies:

pip install -r requirements.txt

## Start API

python -m uvicorn api.main:app --reload

## Swagger

http://127.0.0.1:8000/docs

## API Endpoints

GET /

GET /health

POST /predict

POST /predict/multiview

GET /inspections/{inspection_id}

## Multi-view Images

The multi-view endpoint expects:

front
back
left
right
top
bottom

## Important

The model currently supports mango maturity classification only.

Low-confidence predictions require manual review.

The system is a prototype and requires independent field validation before production use.
