from fastapi import APIRouter, Request

from app.schemas import FamousPlayer, FamousPlayersResponse

router = APIRouter()


@router.get('/players/famous', response_model=FamousPlayersResponse)
def famous_players(request: Request) -> FamousPlayersResponse:
    svc = request.app.state.model_service
    rows = svc.predict_famous()
    return FamousPlayersResponse(
        players=[FamousPlayer(**r) for r in rows],
        feature_names=svc.feature_names,
    )
