from fastapi import APIRouter, Request

from app.schemas import HealthResponse

router = APIRouter()


@router.get('/health', response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    svc = request.app.state.model_service
    return HealthResponse(
        status='ok',
        model_loaded=svc.model is not None,
        feature_names=svc.feature_names,
        metrics=svc.metrics,
        training_meta=svc.training_meta,
        baseline_length=svc.baseline_length,
    )
