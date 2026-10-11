"""
BGTMS - Production FastAPI Backend Service (M1-25)
Exposes detection, tracking, traffic intelligence, predictive AI, incident detection,
and adaptive signal control via async REST and WebSockets.
"""

import os
import io
import time
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response, FileResponse

from bgtms.pipeline import TrafficIntelligencePipeline
from bgtms.core.types import (
    BoundingBox, TrackedVehicle, ApproachState, IncidentAlert, SignalPlan
)
from bgtms.core.config import get_settings, get_classes

app = FastAPI(
    title="BGTMS - Bengaluru Traffic Management System API",
    description="AI-powered traffic intelligence, microscopic density analysis, incident alerts, and adaptive signal optimization.",
    version="1.0.0"
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


@app.on_event("startup")
async def startup_event():
    load_or_create_sample_frames()
    print("[+] BGTMS Traffic Intelligence Service started successfully")


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


@app.get("/api/v1/signal/plan")
async def get_signal_plan():
    """Returns active Webster signal timing plan."""
    return pipeline.latest_signal_plan.model_dump()


@app.get("/api/v1/simulation/benchmark")
async def get_sumo_benchmark():
    """Returns SUMO simulation benchmark comparing Fixed vs BGTMS Adaptive."""
    return pipeline.sumo_simulator.benchmark_comparison()


# Mount static dashboard directory
dashboard_dir = Path(__file__).resolve().parent.parent / "dashboard"
if dashboard_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")


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

            # Process through pipeline
            telemetry = pipeline.process_frame(frame)
            await websocket.send_json(telemetry)
            await asyncio.sleep(0.1) # 10 Hz broadcast
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[!] WebSocket exception: {e}")
