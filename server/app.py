"""
BGTMS - Production FastAPI Backend Service (M1-25)
Exposes detection, tracking, traffic intelligence, predictive AI, incident detection,
and adaptive signal control via async REST and WebSockets.
"""

import sys
import os
import io
import time
import base64
import tempfile
import asyncio

# Fix Windows Proactor IOCP socket aborts (WinError 64: 'The specified network name is no longer available')
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from typing import List, Dict, Any, Optional
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response, FileResponse, RedirectResponse

from bgtms.pipeline import TrafficIntelligencePipeline
from bgtms.core.types import (
    BoundingBox, TrackedVehicle, ApproachState, IncidentAlert, SignalPlan, IncidentType
)

from bgtms.core.config import get_settings, get_classes
from bgtms.mlops.learning_store import learning_store
from bgtms.mlops.auto_retrainer import retrainer

# Synthetic frame cache for simulation stream
_sample_frames: List[np.ndarray] = []


def load_or_create_sample_frames():
    global _sample_frames
    img_dir = Path("data/processed/images")
    if img_dir.exists():
        files = list(img_dir.glob("*.jpg"))[:15]
        for f in files:
            img = cv2.imread(str(f))
            if img is not None:
                _sample_frames.append(img)
    
    if not _sample_frames:
        # Create a synthetic 1920x1080 traffic scene
        blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Road asphalt
        cv2.rectangle(blank, (0, 400), (1920, 1080), (45, 48, 52), -1)
        # Road lines
        for y in range(450, 1080, 80):
            cv2.line(blank, (960, y), (960, y + 40), (255, 255, 255), 4)
        _sample_frames.append(blank)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_or_create_sample_frames()
    print("[+] BGTMS Traffic Intelligence Service started successfully")
    yield


app = FastAPI(
    title="BGTMS - Bengaluru Traffic Management System API",
    description="AI-powered traffic intelligence, microscopic density analysis, incident alerts, and adaptive signal optimization.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize pipeline instance
pipeline = TrafficIntelligencePipeline(
    approach_id="North_HosurRd",
    approach_name="Central Silk Board (Hosur Rd Approach)"
)


@app.get("/")
async def root():
    """Serves the React Vite AI command center directly or redirects to legacy dashboard."""
    frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    index_html = frontend_dist / "index.html"
    if index_html.exists():
        return FileResponse(str(index_html))
    return RedirectResponse(url="/dashboard/")


@app.get("/favicon.ico")
@app.get("/favicon.svg")
async def favicon():
    frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    fav = frontend_dist / "favicon.svg"
    if fav.exists():
        return FileResponse(str(fav), media_type="image/svg+xml")
    return Response(status_code=204)




@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "BGTMS-AI",
        "junction": "BLR-SILKBOARD-01",
        "timestamp": time.time(),
        "version": "1.0.0"
    }


@app.get("/api/v1/classes")
async def get_taxonomy():
    return get_classes()


@app.post("/api/v1/detect")
async def detect_vehicles(file: UploadFile = File(...)):
    """Runs vehicle detection on uploaded image."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image payload")

    detections = pipeline.detector.detect(image)
    return {
        "detections_count": len(detections),
        "detections": [d.model_dump() for d in detections]
    }


CLASS_COLORS = {
    "car": (246, 130, 59),          # Blue/Cyan
    "motorcycle": (16, 185, 129),    # Emerald
    "auto_rickshaw": (11, 158, 245), # Amber
    "bus": (246, 92, 139),          # Purple
    "truck": (199, 116, 56),        # Indigo
    "bicycle": (20, 184, 166),      # Teal
    "pedestrian": (236, 72, 153),   # Pink
    "emergency_vehicle": (68, 68, 239) # Crimson Red
}

PCU_BY_NAME = {
    "car": 1.0,
    "motorcycle": 0.5,
    "auto_rickshaw": 1.0,
    "bus": 3.0,
    "truck": 3.0,
    "bicycle": 0.4,
    "pedestrian": 0.2,
    "emergency_vehicle": 1.5
}


def annotate_image(image: np.ndarray, detections: List[BoundingBox]) -> np.ndarray:
    annotated = image.copy()
    for d in detections:
        color = CLASS_COLORS.get(d.class_name, (200, 200, 200))
        cv2.rectangle(annotated, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), color, 2)
        label = f"{d.class_name} {int(d.confidence * 100)}%"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (int(d.x1), max(0, int(d.y1) - h - 6)), (int(d.x1) + w + 6, int(d.y1)), color, -1)
        cv2.putText(annotated, label, (int(d.x1) + 3, max(h, int(d.y1) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


@app.post("/api/v1/analyze/photo")
async def analyze_photo(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.35, ge=0.05, le=0.95),
    is_red_light: bool = Query(False),
    enable_sahi: bool = Query(True)
):
    """
    Comprehensive single-photo analysis: runs vehicle detection, IRC PCU density,
    congestion classification, incident checking, and logs to Continuous Learning DB + S3.
    """
    start_t = time.time()
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image payload")

    h, w = image.shape[:2]
    detections = pipeline.detector.detect(image, conf_threshold=confidence_threshold, enable_sahi=enable_sahi)
    pipeline_result = pipeline.process_frame(
        image,
        is_red_light=is_red_light,
        conf_threshold=confidence_threshold,
        is_single_frame=True
    )

    class_counts: Dict[str, int] = {}
    total_pcu = 0.0
    for d in detections:
        class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1
        pcu_w = PCU_BY_NAME.get(d.class_name, 1.0)
        total_pcu += pcu_w

    # Continuous Learning Ingestion: Store capture in SQLite DB & S3
    learning_meta = {}
    try:
        learning_meta = learning_store.record_capture(
            image=image,
            filename=file.filename or "photo.jpg",
            media_type="photo",
            detections=detections,
            class_counts=class_counts,
            total_pcu=round(total_pcu, 2)
        )
    except Exception as e:
        print(f"[ContinuousLearning] Notice: {e}")

    annotated = annotate_image(image, detections)
    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_image = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

    elapsed_ms = (time.time() - start_t) * 1000.0

    return {
        "status": "success",
        "filename": file.filename,
        "width": w,
        "height": h,
        "detections_count": len(detections),
        "total_pcu": round(total_pcu, 2),
        "congestion_level": pipeline_result["approach_state"]["congestion_level"],
        "average_speed_kmh": pipeline_result["approach_state"]["average_speed_kmh"],
        "class_breakdown": class_counts,
        "detections": [d.model_dump() for d in detections],
        "approach_state": pipeline_result["approach_state"],
        "incidents": pipeline_result.get("new_incidents", []),
        "signal_plan": pipeline_result["signal_plan"],
        "annotated_image": b64_image,
        "continuous_learning": learning_meta,
        "latency_ms": round(elapsed_ms, 1)
    }


@app.post("/api/v1/analyze/video")
async def analyze_video(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.35, ge=0.05, le=0.95),
    max_seconds: int = Query(30, ge=3, le=120)
):
    """
    High-Throughput Video Analysis: decodes frames, runs fast single-pass tracking,
    aggregates unique traffic volume and PCU, and archives peak traffic frames to Continuous Learning store.
    """
    start_t = time.time()

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    # Cache latest analyzed video for frontend video playback fallback
    try:
        latest_video_path = Path("data/latest_analyzed_video.mp4")
        latest_video_path.parent.mkdir(parents=True, exist_ok=True)
        with open(latest_video_path, "wb") as f_out:
            f_out.write(content)
    except Exception as e:
        print(f"[VideoArchive] Notice: {e}")

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Could not open video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / max(1.0, fps)

    # High-Throughput Optimization: Sample every 0.5s (2 fps) with fast single-pass inference (no tile overhead)
    step = max(1, int(fps * 0.5))
    max_frames_to_process = int(min(duration_sec, max_seconds) * fps)

    unique_tracks: Dict[int, str] = {}
    timeline: List[Dict[str, Any]] = []
    video_incidents: List[Dict[str, Any]] = []
    peak_pcu = 0.0
    peak_frame_b64 = None
    peak_raw_frame = None
    peak_detections = []

    frame_idx = 0
    while cap.isOpened() and frame_idx < max_frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            current_sec = round(frame_idx / fps, 1)
            # Ultra-fast single-pass detection for real-time video tracking
            detections = pipeline.detector.detect(frame, conf_threshold=confidence_threshold, enable_sahi=False)
            tracked = pipeline.tracker.update(detections)

            frame_pcu = 0.0
            for t in tracked:
                unique_tracks[t.track_id] = t.class_name
                frame_pcu += t.pcu

            acc = pipeline.accident_detector.check_incidents(tracked)
            if acc:
                for a in acc:
                    alert_dict = a.model_dump()
                    alert_dict["video_time_sec"] = current_sec
                    video_incidents.append(alert_dict)

            timeline.append({
                "second": current_sec,
                "vehicles": len(tracked),
                "pcu": round(frame_pcu, 2)
            })

            if frame_pcu >= peak_pcu or peak_frame_b64 is None:
                peak_pcu = frame_pcu
                peak_raw_frame = frame.copy()
                peak_detections = detections
                annotated = annotate_image(frame, detections)
                _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                peak_frame_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

        frame_idx += 1

    cap.release()
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    class_breakdown: Dict[str, int] = {}
    total_unique_pcu = 0.0
    for _, cls_name in unique_tracks.items():
        class_breakdown[cls_name] = class_breakdown.get(cls_name, 0) + 1
        total_unique_pcu += PCU_BY_NAME.get(cls_name, 1.0)

    # Ingest peak video scene into Continuous Learning Store
    learning_meta = {}
    if peak_raw_frame is not None:
        try:
            learning_meta = learning_store.record_capture(
                image=peak_raw_frame,
                filename=file.filename or "video.mp4",
                media_type="video",
                detections=peak_detections,
                class_counts=class_breakdown,
                total_pcu=round(peak_pcu, 2)
            )
        except Exception as e:
            print(f"[ContinuousLearning] Video capture notice: {e}")

    elapsed_ms = (time.time() - start_t) * 1000.0

    return {
        "status": "success",
        "filename": file.filename,
        "duration_sec": round(duration_sec, 1),
        "processed_frames": len(timeline),
        "total_unique_vehicles": len(unique_tracks),
        "total_unique_pcu": round(total_unique_pcu, 2),
        "peak_pcu": round(peak_pcu, 2),
        "class_breakdown": class_breakdown,
        "timeline": timeline,
        "incidents": video_incidents,
        "peak_snapshot": peak_frame_b64,
        "continuous_learning": learning_meta,
        "latency_ms": round(elapsed_ms, 1)
    }


@app.get("/api/v1/learning/stats")
async def get_continuous_learning_stats():
    """Returns continuous learning dataset ingestion, S3 sync status, and retraining readiness."""
    return learning_store.get_learning_stats()


@app.post("/api/v1/learning/retrain")
async def trigger_autonomous_retraining(epochs: int = Query(1, ge=1, le=5)):
    """Triggers autonomous model retraining on accumulated SQLite/S3 samples with Experience Replay."""
    result = retrainer.run_retraining(epochs=epochs, force=True)
    if result.get("weights_promoted"):
        pipeline.detector.reload_model_weights()
    return result


@app.get("/api/v1/sample/photo")
async def get_sample_photo():
    """Serves a pre-loaded Indian traffic sample image."""
    img_dir = Path("data/processed/images")
    if img_dir.exists():
        files = list(img_dir.glob("*.jpg"))
        if files:
            return FileResponse(str(files[0]), media_type="image/jpeg")
    if _sample_frames:
        _, buffer = cv2.imencode(".jpg", _sample_frames[0])
        return Response(content=buffer.tobytes(), media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="No sample photos available")


@app.get("/api/v1/sample/video")
async def get_sample_video():
    """Serves the generated sample traffic video."""
    video_path = Path("data/sample_traffic.mp4")
    if video_path.exists():
        return FileResponse(str(video_path), media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Sample video not found")


@app.get("/api/v1/video/latest")
async def get_latest_analyzed_video():
    """Serves the most recently analyzed video file for frontend playback fallback."""
    latest_video_path = Path("data/latest_analyzed_video.mp4")
    if latest_video_path.exists():
        return FileResponse(str(latest_video_path), media_type="video/mp4")
    sample_path = Path("data/sample_traffic.mp4")
    if sample_path.exists():
        return FileResponse(str(sample_path), media_type="video/mp4")
    raise HTTPException(status_code=404, detail="No video available")


@app.post("/api/v1/process-frame")
async def process_frame_end_to_end(file: UploadFile = File(...), is_red_light: bool = False):
    """Processes frame through full end-to-end intelligence pipeline."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image payload")

    result = pipeline.process_frame(image, is_red_light=is_red_light)
    return result


@app.get("/api/v1/traffic/state")
async def get_traffic_state():
    """Returns real-time microscopic traffic density and queue telemetry."""
    return {
        "junction_id": "BLR-SILKBOARD-01",
        "approach": pipeline.latest_approach_state.model_dump(),
        "signal_plan": pipeline.latest_signal_plan.model_dump(),
        "timestamp": time.time()
    }


@app.get("/api/v1/traffic/predict")
async def get_congestion_predictions():
    """Returns 5, 15, and 30-minute congestion forecasts."""
    return {
        "junction_id": "BLR-SILKBOARD-01",
        "forecasts": pipeline.predictor.predict(),
        "timestamp": time.time()
    }


@app.get("/api/v1/incidents")
async def get_incident_alerts(limit: int = Query(20, ge=1, le=100)):
    """Returns recent incidents, accidents, and violation alerts."""
    alerts = pipeline.all_incidents[-limit:]
    return {
        "total_incidents": len(pipeline.all_incidents),
        "alerts": [a.model_dump() for a in reversed(alerts)]
    }


@app.post("/api/v1/incidents/emergency-override")
async def trigger_emergency_green_corridor(approach_id: str = "North_HosurRd", enable: bool = True):
    """Manually triggers or clears emergency Green Corridor preemption."""
    target = approach_id if enable else None
    plan = pipeline.signal_optimizer.compute_optimal_plan(
        approaches={pipeline.approach_id: pipeline.latest_approach_state},
        green_corridor_approach=target
    )
    pipeline.latest_signal_plan = plan
    return {
        "status": "GREEN_CORRIDOR_ACTIVE" if enable else "NORMAL_ADAPTIVE",
        "target_approach": approach_id,
        "signal_plan": plan.model_dump()
    }


@app.post("/api/v1/incidents/simulate")
async def simulate_incident(incident_type: str = "COLLISION", approach_id: str = "North_HosurRd"):
    """Simulates an incident for real-time testing on the dashboard and operator consoles."""
    import uuid
    alert_type = IncidentType.COLLISION
    desc = f"Collision detected between two vehicles on {approach_id}"
    sev = "CRITICAL"

    if incident_type.upper() == "EMERGENCY":
        alert_type = IncidentType.EMERGENCY_VEHICLE_APPROACH
        desc = f"Ambulance #108 approaching on {approach_id}. Green corridor activated."
        sev = "CRITICAL"
    elif incident_type.upper() == "WRONG_WAY":
        alert_type = IncidentType.WRONG_WAY
        desc = f"Dangerous Wrong-Way driver detected on {approach_id}"
        sev = "CRITICAL"
    elif incident_type.upper() == "STALL":
        alert_type = IncidentType.STALLED_VEHICLE
        desc = f"Stalled Bus blocking central lane on {approach_id}"
        sev = "WARNING"

    alert = IncidentAlert(
        incident_id=f"SIM-{uuid.uuid4().hex[:6].upper()}",
        incident_type=alert_type,
        severity=sev,
        timestamp=time.time(),
        track_id=999,
        approach_id=approach_id,
        location=(640.0, 360.0),
        description=desc,
        details={"simulation": True, "type": incident_type}
    )
    pipeline.all_incidents.append(alert)
    return {"status": "SUCCESS", "incident": alert.model_dump()}


@app.get("/api/v1/signal/plan")
async def get_signal_plan():
    """Returns active Webster signal timing plan."""
    return pipeline.latest_signal_plan.model_dump()


@app.get("/api/v1/model-info")
async def get_model_info():
    """Returns traceable model versions, checkpoint sizes, and configurations (Task BGT-31)."""
    models_dir = Path(__file__).resolve().parent.parent / "bgtms" / "models"
    det_pth = models_dir / "vehicle_detector.pth"
    det_onnx = models_dir / "vehicle_detector.onnx"
    fc_joblib = models_dir / "traffic_forecaster.joblib"
    cls_joblib = models_dir / "congestion_classifier.joblib"

    return {
        "service": "BGTMS-AI",
        "version": "1.0.0",
        "models": {
            "vehicle_detector_pytorch": {
                "architecture": "SSDLite320-MobileNetV3",
                "weights_file": "vehicle_detector.pth",
                "exists": det_pth.exists(),
                "size_bytes": det_pth.stat().st_size if det_pth.exists() else 0,
                "classes": 8
            },
            "vehicle_detector_onnx": {
                "format": "ONNX v14",
                "weights_file": "vehicle_detector.onnx",
                "exists": det_onnx.exists(),
                "size_bytes": det_onnx.stat().st_size if det_onnx.exists() else 0
            },
            "traffic_forecaster": {
                "algorithm": "RandomForestRegressor / GBR",
                "weights_file": "traffic_forecaster.joblib",
                "exists": fc_joblib.exists(),
                "size_bytes": fc_joblib.stat().st_size if fc_joblib.exists() else 0
            },
            "congestion_classifier": {
                "algorithm": "RandomForestClassifier",
                "weights_file": "congestion_classifier.joblib",
                "exists": cls_joblib.exists(),
                "size_bytes": cls_joblib.stat().st_size if cls_joblib.exists() else 0
            }
        },
        "configs": get_settings()
    }


@app.get("/api/v1/simulation/benchmark")
async def get_sumo_benchmark():
    """Returns SUMO simulation benchmark comparing Fixed vs BGTMS Adaptive."""
    return pipeline.sumo_simulator.benchmark_comparison()


# Mount static dashboard directory and React frontend build
dashboard_dir = Path(__file__).resolve().parent.parent / "dashboard"
if dashboard_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")

frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    app.mount("/app", StaticFiles(directory=str(frontend_dist), html=True), name="react-app")


@app.websocket("/ws/live-telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """
    High-frequency WebSocket broadcasting live CCTV telemetry, tracks, and signals.
    """
    await websocket.accept()
    frame_idx = 0
    try:
        while True:
            # Pick next frame from cyclic buffer
            frame = _sample_frames[frame_idx % len(_sample_frames)]
            frame_idx += 1

            # Process in background thread to never block the asyncio event loop or HTTP endpoints
            telemetry = await asyncio.to_thread(pipeline.process_frame, frame)
            await websocket.send_json(telemetry)
            await asyncio.sleep(1.0) # 1 Hz broadcast (responsive, lightweight)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[!] WebSocket exception: {e}")
