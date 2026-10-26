"""QuantumSentinel FastAPI application."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from app.config import (
    API_VERSION,
    CORS_ORIGIN_REGEX,
    CORS_ORIGINS,
    DATABASE_CONNECT_TIMEOUT,
    DATABASE_POOL_MAX_SIZE,
    DATABASE_POOL_MIN_SIZE,
    DATABASE_URL,
    FRONTEND_FILE,
    MODEL_FILE,
    STANDARD_QBER_THRESHOLD,
)
from app.db import (
    PostgresSessionRepository,
    SessionRecord,
    SessionStoreError,
)
from app.detection.detector import Detector, ModelLoadError, load_detector
from app.detection.features import FEATURE_NAMES, extract_features
from app.schemas import (
    HealthResponse,
    RunRequest,
    RunResponse,
    SessionDetail,
    SessionListResponse,
    SessionSummary,
)
from app.simulation.bb84 import (
    Engine,
    EveMode,
    SimulationRequest,
    simulate_session,
)
from app.simulation.channels import (
    ChannelType,
    QiskitUnavailableError,
    qiskit_available,
)

logger = logging.getLogger("quantumsentinel")


def _detector(request: Request) -> Detector:
    detector = getattr(request.app.state, "detector", None)
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Detector model is not loaded. From the backend directory run "
                "`python -m app.detection.train`, then restart the API."
            ),
        )
    return detector


def _model_summary(detector: Detector) -> dict[str, Any]:
    return detector.summary()


def _repository(request: Request) -> PostgresSessionRepository:
    repository = getattr(request.app.state, "session_repository", None)
    if repository is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Session archive is not configured. Set QS_DATABASE_URL to a "
                "PostgreSQL database and restart the API."
            ),
        )
    return repository


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    app.state.detector = None
    app.state.session_repository = None

    try:
        app.state.detector = load_detector(MODEL_FILE)
        app.state.model_metadata = app.state.detector.metadata
        logger.info("Detector ready: %s", MODEL_FILE)
    except ModelLoadError as exc:
        app.state.model_metadata = None
        logger.error("Detector unavailable: %s", exc)
        logger.error(
            "The API will serve the dashboard and /health, but /run returns 503 "
            "until the model is trained."
        )

    try:
        repository = PostgresSessionRepository(
            DATABASE_URL,
            min_size=DATABASE_POOL_MIN_SIZE,
            max_size=DATABASE_POOL_MAX_SIZE,
            connect_timeout=DATABASE_CONNECT_TIMEOUT,
        )
        app.state.session_repository = repository
        try:
            repository.initialize()
            logger.info(
                "Session archive ready: postgresql database %s",
                repository.database_name or "(unknown)",
            )
        except SessionStoreError as exc:
            logger.error("Session archive unavailable: %s", exc)
            logger.error(
                "The detector is still available, but /run and /sessions return "
                "503 until PostgreSQL is reachable."
            )
    except SessionStoreError as exc:  # pragma: no cover - empty DSN only
        logger.error("Session archive is misconfigured: %s", exc)

    logger.info("Qiskit engine available: %s", qiskit_available())
    try:
        yield
    finally:
        repository = getattr(app.state, "session_repository", None)
        if repository is not None:
            repository.close()


app = FastAPI(
    title="QuantumSentinel API",
    summary="BB84 simulation and ML intrusion detection",
    description=(
        "Simulates BB84 quantum key distribution sessions with configurable "
        "noise and eavesdropping, classifies each session with a trained "
        "intrusion detector, and archives completed runs in PostgreSQL. The "
        "detector never receives eavesdropper ground truth; labels are exposed "
        "separately for simulator diagnostics only."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time-Ms"],
)


@app.middleware("http")
async def timing_and_cache_headers(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    if request.url.path.startswith(("/run", "/sessions", "/health", "/model", "/api/")):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/", include_in_schema=False)
def dashboard():
    if FRONTEND_FILE.exists():
        return FileResponse(
            FRONTEND_FILE,
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )
    return JSONResponse(
        {
            "name": "QuantumSentinel API",
            "version": API_VERSION,
            "detail": (
                f"Dashboard not found at {FRONTEND_FILE}. The API is available "
                "at /docs."
            ),
        }
    )


@app.get("/index.html", include_in_schema=False)
def dashboard_index():
    return dashboard()


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    detector: Detector | None = getattr(request.app.state, "detector", None)
    repository: PostgresSessionRepository | None = getattr(
        request.app.state, "session_repository", None
    )
    database_connected = repository.ping() if repository is not None else False
    healthy = detector is not None and database_connected
    return HealthResponse(
        status="ok" if healthy else "degraded",
        version=API_VERSION,
        model_loaded=detector is not None,
        qiskit_available=qiskit_available(),
        database_connected=database_connected,
        database_name=repository.database_name if repository is not None else None,
        model=detector.summary() if detector is not None else None,
    )


@app.get("/model", tags=["system"])
def model_card(request: Request) -> JSONResponse:
    detector = _detector(request)
    metadata = getattr(request.app.state, "model_metadata", detector.metadata)
    return JSONResponse(jsonable_encoder(metadata))


@app.post("/run", response_model=RunResponse, tags=["simulation"])
def run_session(payload: RunRequest, request: Request) -> RunResponse:
    """Run one BB84 session and classify it.

    `standard_detected` applies the traditional 11% QBER rule.
    `ml_detected` comes exclusively from the trained classifier.  Simulator
    ground truth is returned separately in `attack_present` / `attack_mode`
    so the demo can highlight threshold-rule evasion.
    """
    detector = _detector(request)
    repository = _repository(request)
    started = time.perf_counter()

    try:
        simulation_request = SimulationRequest(
            eve=EveMode(payload.eve),
            eve_strength=float(payload.eve_strength),
            noise=float(payload.noise),
            qubits=int(payload.qubits),
            channel=ChannelType(payload.channel),
            engine=Engine(payload.engine),
            seed=payload.seed,
        )
        result = simulate_session(simulation_request)
        features = extract_features(result.observation)
        detection = detector.predict_features(features)
    except QiskitUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        logger.exception("Session processing failed")
        raise HTTPException(
            status_code=500, detail="Session processing failed on the server."
        )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    observation = result.observation
    standard_detected = observation.qber > STANDARD_QBER_THRESHOLD
    session_id = str(uuid.uuid4())
    model = _model_summary(detector)
    feature_map = {
        name: float(value) for name, value in zip(FEATURE_NAMES, features)
    }

    record = SessionRecord(
        session_id=session_id,
        engine=result.engine_used,
        engine_requested=payload.engine,
        eve_mode=payload.eve,
        eve_strength=int(payload.eve_strength),
        noise=int(payload.noise),
        qubits=int(simulation_request.qubits),
        channel=payload.channel,
        seed=result.seed,
        qber=float(observation.qber),
        threshold=STANDARD_QBER_THRESHOLD,
        standard_detected=standard_detected,
        ml_detected=detection.detected,
        ml_confidence=float(detection.confidence * 100.0),
        attack_probability=float(detection.attack_probability),
        attack_present=result.attack_present,
        attack_mode=result.attack_mode,
        block_qber=[float(value) for value in observation.block_qber],
        sifted_bits=int(observation.sifted_bits),
        sifted_rate=float(observation.sifted_rate),
        x_qber=float(observation.x_qber),
        z_qber=float(observation.z_qber),
        expected_qber=float(observation.expected_qber),
        elapsed_ms=float(elapsed_ms),
        model_name=str(model["name"]),
        model_version=str(model["version"]),
        model_threshold=float(model["threshold"]),
        features=feature_map,
    )

    try:
        repository.save(record)
    except SessionStoreError:
        logger.exception("Could not archive session %s", session_id)
        raise HTTPException(
            status_code=503,
            detail=(
                "The session completed but could not be archived. "
                "PostgreSQL is unavailable; check /health and the API log."
            ),
        )

    logger.info(
        "session=%s engine=%s eve=%s noise=%.1f qubits=%d qber=%.2f "
        "standard=%s ml=%s p=%.3f in %.1fms archived=true",
        session_id,
        result.engine_used,
        result.attack_mode,
        observation.noise_percent,
        simulation_request.qubits,
        observation.qber,
        standard_detected,
        detection.detected,
        detection.attack_probability,
        elapsed_ms,
    )

    return RunResponse(
        session_id=session_id,
        engine=result.engine_used,  # type: ignore[arg-type]
        engine_requested=payload.engine,
        warning=None,
        qber=round(observation.qber, 2),
        threshold=STANDARD_QBER_THRESHOLD,
        standard_detected=standard_detected,
        ml_detected=detection.detected,
        ml_confidence=round(detection.confidence * 100.0, 2),
        attack_probability=round(detection.attack_probability, 4),
        attack_present=result.attack_present,
        attack_mode=result.attack_mode,  # type: ignore[arg-type]
        block_qber=[round(value, 2) for value in observation.block_qber],
        sifted_bits=observation.sifted_bits,
        sifted_rate=round(observation.sifted_rate, 4),
        x_qber=round(observation.x_qber, 2),
        z_qber=round(observation.z_qber, 2),
        expected_qber=round(observation.expected_qber, 2),
        elapsed_ms=round(elapsed_ms, 2),
        seed=result.seed,
        model=model,
    )


@app.get("/sessions", response_model=SessionListResponse, tags=["sessions"])
def list_sessions(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    ml_detected: bool | None = Query(
        None, description="Filter by the ML verdict."
    ),
) -> SessionListResponse:
    """List archived sessions, newest first."""
    repository = _repository(request)
    try:
        rows, total = repository.list_sessions(
            limit=limit, offset=offset, ml_detected=ml_detected
        )
    except SessionStoreError as exc:
        logger.error("Could not read session archive: %s", exc)
        raise HTTPException(
            status_code=503, detail="Session archive is unavailable."
        ) from exc

    return SessionListResponse(
        items=[SessionSummary(**row) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@app.get(
    "/sessions/{session_id}",
    response_model=SessionDetail,
    tags=["sessions"],
)
def get_session(session_id: UUID, request: Request) -> SessionDetail:
    """Return the full archived record for one session."""
    repository = _repository(request)
    try:
        row = repository.get_session(str(session_id))
    except SessionStoreError as exc:
        logger.error("Could not read archived session %s: %s", session_id, exc)
        raise HTTPException(
            status_code=503, detail="Session archive is unavailable."
        ) from exc
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return SessionDetail(**row)


@app.post("/api/v1/run", response_model=RunResponse, include_in_schema=False)
def run_session_v1(payload: RunRequest, request: Request) -> RunResponse:
    return run_session(payload, request)


@app.get("/api/v1/health", response_model=HealthResponse, include_in_schema=False)
def health_v1(request: Request) -> HealthResponse:
    return health(request)


@app.get(
    "/api/v1/sessions",
    response_model=SessionListResponse,
    include_in_schema=False,
)
def list_sessions_v1(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    ml_detected: bool | None = Query(None),
) -> SessionListResponse:
    return list_sessions(request, limit, offset, ml_detected)


@app.get(
    "/api/v1/sessions/{session_id}",
    response_model=SessionDetail,
    include_in_schema=False,
)
def get_session_v1(session_id: UUID, request: Request) -> SessionDetail:
    return get_session(session_id, request)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
