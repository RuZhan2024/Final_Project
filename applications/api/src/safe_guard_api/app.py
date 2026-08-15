from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from safe_guard_api.converters import prediction_to_response, request_to_window
from safe_guard_api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    RuntimeResponse,
)
from safe_guard_api.settings import ApiSettings
from safe_guard_ml import PredictionRuntime


def create_app(
    runtime: PredictionRuntime | None = None,
    settings: ApiSettings | None = None,
) -> FastAPI:
    resolved_settings = settings or ApiSettings.from_env()
    resolved_runtime = runtime or PredictionRuntime.baseline()

    app = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["content-type"],
    )

    @app.exception_handler(ValueError)
    async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=resolved_settings.app_name,
            version=resolved_settings.app_version,
        )

    @app.get("/api/v1/runtime", response_model=RuntimeResponse)
    def runtime_info() -> RuntimeResponse:
        return RuntimeResponse(
            model_name=resolved_runtime.predictor.model_name,
            input_type="skeleton_window",
        )

    @app.post("/api/v1/predictions", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        prediction = resolved_runtime.predict(request_to_window(request))
        return prediction_to_response(prediction)

    return app


app = create_app()
