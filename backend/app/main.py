"""
FastAPI 진입점 — uvicorn target.

실행:
    cd backend
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_health, routes_predict, routes_players
from app.config import CORS_ORIGINS, MODEL_PATH
from app.services.model_service import ModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model once
    svc = ModelService(MODEL_PATH)
    svc.load()
    app.state.model_service = svc
    print(f"✓ Model loaded from {MODEL_PATH} | features: {svc.feature_names}")
    yield
    # Shutdown: nothing to clean up; TF graph released at process exit


app = FastAPI(title='NFL Survival API', version='0.1.0', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(routes_health.router, prefix='/api')
app.include_router(routes_predict.router, prefix='/api')
app.include_router(routes_players.router, prefix='/api')


@app.get('/')
def root():
    return {'service': 'nfl-survival', 'docs': '/docs', 'health': '/api/health'}
