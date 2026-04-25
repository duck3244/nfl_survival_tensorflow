from fastapi import APIRouter, HTTPException, Request

from app.schemas import PredictRequest, PredictResponse

router = APIRouter()


@router.post('/predict', response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    svc = request.app.state.model_service
    try:
        result = svc.predict(payload.features, payload.max_games)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f'Prediction failed: {e}')
    return PredictResponse(**result)
