"""
BGTMS - FastAPI Backend Endpoint & Schema Validation Tests (Task BGT-31)
Tests health, taxonomy, model info, detection, prediction, incidents, and signal APIs.
"""

import os
import sys
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from server.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "BGTMS-AI"
    assert data["junction"] == "BLR-SILKBOARD-01"
    assert "timestamp" in data


def test_classes_taxonomy_endpoint():
    response = client.get("/api/v1/classes")
    assert response.status_code == 200
    data = response.json()
    assert "classes" in data
    classes = data["classes"]
    assert len(classes) >= 8
    # Check Indian PCU weights exist
    assert "pcu" in classes["0"] or "pcu" in classes[0]


def test_model_info_traceability():
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "vehicle_detector_pytorch" in data["models"]
    assert "vehicle_detector_onnx" in data["models"]
    assert "traffic_forecaster" in data["models"]
    assert "congestion_classifier" in data["models"]
    assert "configs" in data


def test_detect_endpoint_with_image():
    # Create 320x320 RGB dummy image
    img = Image.new("RGB", (320, 320), color=(120, 150, 180))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/v1/detect",
        files={"file": ("test.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "detections_count" in data
    assert "detections" in data
    assert isinstance(data["detections"], list)


def test_process_frame_end_to_end():
    img = Image.new("RGB", (320, 320), color=(100, 110, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/v1/process-frame?is_red_light=false",
        files={"file": ("frame.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "frame_index" in data
    assert "approach_state" in data
    assert "signal_plan" in data


def test_traffic_state_endpoint():
    response = client.get("/api/v1/traffic/state")
    assert response.status_code == 200
    data = response.json()
    assert "junction_id" in data
    assert "approach" in data
    assert "signal_plan" in data


def test_traffic_predict_endpoint():
    response = client.get("/api/v1/traffic/predict")
    assert response.status_code == 200
    data = response.json()
    assert "forecasts" in data
    forecasts = data["forecasts"]
    assert "5_min" in forecasts or "15_min" in forecasts or len(forecasts) >= 1


def test_incidents_endpoint():
    response = client.get("/api/v1/incidents?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "total_incidents" in data
    assert "alerts" in data
    assert isinstance(data["alerts"], list)


def test_emergency_override_endpoint():
    response = client.post("/api/v1/incidents/emergency-override?approach_id=North_HosurRd&enable=true")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "GREEN_CORRIDOR_ACTIVE"
    assert data["signal_plan"]["is_green_corridor"] is True


def test_signal_plan_endpoint():
    response = client.get("/api/v1/signal/plan")
    assert response.status_code == 200
    data = response.json()
    assert "phases" in data
    assert "cycle_time_sec" in data


def test_root_redirect():
    response = client.get("/", follow_redirects=False)
    assert response.status_code in [307, 302, 308]
    assert response.headers["location"] == "/dashboard/"


def test_simulate_incident_endpoint():
    response = client.post("/api/v1/incidents/simulate?incident_type=COLLISION&approach_id=North_HosurRd")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "incident" in data
    assert data["incident"]["incident_type"] == "COLLISION"
