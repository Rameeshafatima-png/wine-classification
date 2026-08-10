import joblib
import numpy as np
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel, Field


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "model_bundle.pkl"

TEMPLATES_DIR = BASE_DIR / "templates"

STATIC_DIR = BASE_DIR / "static"


# ==========================================
# CHECK MODEL
# ==========================================

if not MODEL_PATH.exists():

    raise RuntimeError(
        "model_bundle.pkl not found! "
        "Please run train.py first."
    )


# ==========================================
# LOAD MODEL
# ==========================================

model_bundle = joblib.load(
    MODEL_PATH
)

model = model_bundle["model"]

target_names = model_bundle["target_names"]

accuracy = model_bundle["accuracy"]

version = model_bundle["version"]

features = model_bundle["features"]


# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(
    title="WineVision AI",
    description="Wine Classification using Random Forest",
    version=version
)


# ==========================================
# STATIC FILES
# ==========================================

app.mount(
    "/static",
    StaticFiles(
        directory=str(STATIC_DIR)
    ),
    name="static"
)


# ==========================================
# TEMPLATES
# ==========================================

templates = Jinja2Templates(
    directory=str(TEMPLATES_DIR)
)


# ==========================================
# INPUT SCHEMA
# ==========================================

class WineInput(BaseModel):

    alcohol: float = Field(
        ...,
        gt=0
    )

    malic_acid: float = Field(
        ...,
        gt=0
    )

    proline: float = Field(
        ...,
        gt=0
    )


# ==========================================
# HOME PAGE
# ==========================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "model_version": version,
            "model_accuracy": f"{accuracy * 100:.2f}%"
        }
    )


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": version,
        "accuracy": f"{accuracy * 100:.2f}%"
    }


# ==========================================
# PREDICTION
# ==========================================

@app.post("/predict")
def predict(data: WineInput):

    try:

        # ==================================
        # CREATE INPUT ARRAY
        # ==================================

        features_data = np.array([[
            data.alcohol,
            data.malic_acid,
            data.proline
        ]])


        # ==================================
        # PREDICT
        # ==================================

        prediction = int(
            model.predict(features_data)[0]
        )


        # ==================================
        # CONFIDENCE
        # ==================================

        probabilities = (
            model.predict_proba(
                features_data
            )[0]
        )


        confidence = float(
            probabilities[prediction]
        )


        # ==================================
        # RETURN ONLY 3 OUTPUTS
        # ==================================

        return {

            "predicted_class": prediction,

            "predicted_label":
                target_names[prediction],

            "confidence":
                round(
                    confidence * 100,
                    2
                )
        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================
# MODEL INFORMATION
# ==========================================

@app.get("/model-info")
def model_info():

    return {

        "project":
            "Wine Dataset Classification",

        "model":
            "Random Forest",

        "model_version":
            version,

        "accuracy":
            f"{accuracy * 100:.2f}%",

        "features":
            features,

        "classes":
            list(target_names)
    }