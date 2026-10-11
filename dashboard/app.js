// BGTMS Dashboard Client (M1-26)

let isGreenCorridor = false;
let ws = null;
let countdownSec = 38;
let currentPhase = 1;
const canvas = document.getElementById("cctvCanvas");
const ctx = canvas.getContext("2d");

// Simulated active vehicles on canvas for realistic CCTV reproduction
let simVehicles = [
    { id: 1, type: "auto_rickshaw", x: 280, y: 350, speed: 28, lane: 1, w: 42, h: 60, color: "#f59e0b" },
    { id: 2, type: "motorcycle", x: 340, y: 220, speed: 38, lane: 1, w: 24, h: 48, color: "#10b981" },
    { id: 3, type: "car", x: 620, y: 480, speed: 24, lane: 2, w: 56, h: 90, color: "#3b82f6" },
    { id: 4, type: "bus", x: 920, y: 300, speed: 18, lane: 3, w: 75, h: 160, color: "#8b5cf6" },
    { id: 5, type: "motorcycle", x: 580, y: 320, speed: 32, lane: 2, w: 24, h: 48, color: "#10b981" },
    { id: 6, type: "auto_rickshaw", x: 670, y: 200, speed: 26, lane: 2, w: 42, h: 60, color: "#f59e0b" },
    { id: 7, type: "car", x: 930, y: 520, speed: 20, lane: 3, w: 56, h: 90, color: "#3b82f6" },
    { id: 8, type: "ambulance", x: 610, y: 120, speed: 45, lane: 2, w: 60, h: 105, color: "#ef4444", emergency: true }
];

function init() {
    setupCanvas();
    setupWebSocket();
    startRenderLoop();
    startCountdown();
    fetchBenchmark();
}

function setupCanvas() {
    canvas.width = 1280;
    canvas.height = 720;
}

function startRenderLoop() {
    function render() {
        drawCCTVScene();
        requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
}

function drawCCTVScene() {
    const w = canvas.width;
    const h = canvas.height;

    // Background Asphalt
    ctx.fillStyle = "#1e2430";
    ctx.fillRect(0, 0, w, h);

    // Perspective Road Trapezoid
    ctx.fillStyle = "#27303f";
    ctx.beginPath();
    ctx.moveTo(w * 0.15, h);
    ctx.lineTo(w * 0.35, h * 0.15);
    ctx.lineTo(w * 0.75, h * 0.15);
    ctx.lineTo(w * 0.95, h);
    ctx.closePath();
    ctx.fill();

    // Lane Dividers
    ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
    ctx.lineWidth = 3;
    ctx.setLineDash([20, 15]);

    // Lane 1-2 divider
    ctx.beginPath();
    ctx.moveTo(w * 0.42, h);
    ctx.lineTo(w * 0.48, h * 0.15);
    ctx.stroke();

    // Lane 2-3 divider
    ctx.beginPath();
    ctx.moveTo(w * 0.68, h);
    ctx.lineTo(w * 0.62, h * 0.15);
    ctx.stroke();
    ctx.setLineDash([]);

    // Stop Line (Zebra boundary)
    ctx.strokeStyle = currentPhase === 1 ? "#10b981" : "#ef4444";
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.moveTo(w * 0.18, h * 0.85);
    ctx.lineTo(w * 0.92, h * 0.85);
    ctx.stroke();

    // Stop Line Label
    ctx.fillStyle = currentPhase === 1 ? "#10b981" : "#ef4444";
    ctx.font = "bold 13px JetBrains Mono";
    ctx.fillText(`STOP LINE [${currentPhase === 1 ? 'SIGNAL GREEN' : 'SIGNAL RED'}]`, w * 0.2, h * 0.83);

    // Animate & Draw Vehicles
    simVehicles.forEach(v => {
        // Move vehicle downward
        let moveStep = (v.speed / 20.0);
        if (currentPhase === 2 && v.y > h * 0.7 && v.y < h * 0.85) {
            moveStep = 0.2; // stopped in queue
        }
        v.y += moveStep;
        if (v.y > h + 100) {
            v.y = h * 0.15; // wrap around
        }

        // Draw Bounding Box
        ctx.strokeStyle = v.emergency ? "#ef4444" : v.color;
        ctx.lineWidth = v.emergency ? 3 : 2;
        ctx.strokeRect(v.x - v.w / 2, v.y - v.h / 2, v.w, v.h);

        // Fill semi-transparent box
        ctx.fillStyle = v.emergency ? "rgba(239, 68, 68, 0.25)" : "rgba(15, 23, 42, 0.4)";
        ctx.fillRect(v.x - v.w / 2, v.y - v.h / 2, v.w, v.h);

        // Emergency Flashing Beacon
        if (v.emergency) {
            ctx.fillStyle = (Date.now() % 400 < 200) ? "#ef4444" : "#3b82f6";
            ctx.beginPath();
            ctx.arc(v.x, v.y - v.h / 2 - 8, 6, 0, Math.PI * 2);
            ctx.fill();
        }

        // Label Tag
        ctx.fillStyle = v.emergency ? "#ef4444" : v.color;
        const tag = `#${v.id} ${v.type.toUpperCase()} ${v.speed}km/h`;
        ctx.font = "11px JetBrains Mono";
        const tagW = ctx.measureText(tag).width;
        ctx.fillRect(v.x - v.w / 2, v.y - v.h / 2 - 16, tagW + 8, 16);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(tag, v.x - v.w / 2 + 4, v.y - v.h / 2 - 4);
    });
}

function startCountdown() {
    setInterval(() => {
        countdownSec--;
        if (countdownSec <= 0) {
            currentPhase = currentPhase === 1 ? 2 : 1;
            countdownSec = currentPhase === 1 ? 52 : 34;

            const pTag = document.getElementById("phaseTag");
            const dial = document.getElementById("dialCircle");
            if (currentPhase === 1) {
                pTag.textContent = "PHASE 1 • GREEN (N-S)";
                pTag.style.color = "var(--accent-emerald)";
                dial.style.borderColor = "var(--accent-emerald)";
            } else {
                pTag.textContent = "PHASE 2 • GREEN (E-W)";
                pTag.style.color = "var(--accent-amber)";
                dial.style.borderColor = "var(--accent-amber)";
            }
        }
        document.getElementById("dialSec").textContent = countdownSec;
    }, 1000);
}

function setupWebSocket() {
    const loc = window.location;
    const wsUrl = `ws://${loc.hostname || 'localhost'}:${loc.port || '8000'}/ws/live-telemetry`;

    try {
        ws = new WebSocket(wsUrl);
        ws.onopen = () => {
            document.getElementById("linkStatus").textContent = "CONNECTED (WS)";
            document.getElementById("linkStatus").className = "value val-green";
        };
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateDashboardTelemetry(data);
        };
        ws.onerror = () => {
            fallbackPolling();
        };
        ws.onclose = () => {
            fallbackPolling();
        };
    } catch (e) {
        fallbackPolling();
    }
}

function fallbackPolling() {
    document.getElementById("linkStatus").textContent = "POLLING (HTTP)";
    document.getElementById("linkStatus").className = "value val-amber";

    setInterval(async () => {
        try {
            const res = await fetch("/api/v1/traffic/state");
            if (res.ok) {
                const data = await res.json();
                updateFromState(data);
            }
        } catch (e) {}
    }, 2000);
}

function updateDashboardTelemetry(data) {
    if (data.latency_ms) {
        document.getElementById("latencyVal").textContent = `${data.latency_ms} ms (${data.fps} FPS)`;
    }
    if (data.approach_state) {
        const app = data.approach_state;
        document.getElementById("hudPcuCount").textContent = app.total_pcu;
        document.getElementById("hudSpeed").textContent = `${app.average_speed_kmh} km/h`;
        document.getElementById("hudDetCount").textContent = app.total_vehicles;

        // Lane updates
        if (app.lanes && app.lanes.length >= 3) {
            updateLane(1, app.lanes[0]);
            updateLane(2, app.lanes[1]);
            updateLane(3, app.lanes[2]);
        }
    }
    if (data.forecasts) {
        updateForecasts(data.forecasts);
    }
    if (data.new_incidents && data.new_incidents.length > 0) {
        data.new_incidents.forEach(inc => addIncidentAlert(inc));
    }
}

function addIncidentAlert(inc) {
    const feed = document.getElementById("alertsFeed");
    if (!feed) return;
    const item = document.createElement("div");
    const sevClass = inc.severity === "CRITICAL" ? "alert-critical" : (inc.severity === "WARNING" ? "alert-warning" : "alert-info");
    item.className = `alert-item ${sevClass}`;
    item.innerHTML = `
        <div class="alert-meta">
            <span class="alert-type">${(inc.incident_type || 'INCIDENT').replace(/_/g, ' ')}</span>
            <span class="alert-time">JUST NOW</span>
        </div>
        <p class="alert-desc">${inc.description || 'Traffic incident flagged by AI vision'}</p>
    `;
    feed.insertBefore(item, feed.firstChild);
    while (feed.children.length > 15) {
        feed.removeChild(feed.lastChild);
    }
    const countBadge = document.getElementById("alertCount");
    if (countBadge) {
        countBadge.textContent = `${feed.children.length} ALERTS`;
    }
}

async function simulateIncident(type) {
    try {
        const res = await fetch(`/api/v1/incidents/simulate?incident_type=${type}`, { method: "POST" });
        if (res.ok) {
            const data = await res.json();
            if (data.incident) {
                addIncidentAlert(data.incident);
            }
        }
    } catch (e) {
        // Fallback local simulation if offline
        addIncidentAlert({
            incident_type: type === "EMERGENCY" ? "EMERGENCY_VEHICLE" : "COLLISION",
            severity: "CRITICAL",
            description: type === "EMERGENCY" ? "Ambulance #108 approaching central corridor. Preemption triggered." : "Simulated abrupt collision detected on North Approach."
        });
    }
}

function updateFromState(data) {
    if (data.approach) {
        document.getElementById("hudPcuCount").textContent = data.approach.total_pcu;
        document.getElementById("hudSpeed").textContent = `${data.approach.average_speed_kmh} km/h`;
        document.getElementById("hudDetCount").textContent = data.approach.total_vehicles;
    }
}

function updateLane(laneNum, metrics) {
    document.getElementById(`lane${laneNum}Count`).textContent = `${metrics.vehicle_count} veh (${metrics.pcu_count} PCU)`;
    document.getElementById(`lane${laneNum}Bar`).style.width = `${Math.min(100, metrics.occupancy_pct)}%`;
    document.getElementById(`lane${laneNum}Queue`).textContent = `${metrics.queue_length_m}m`;
    document.getElementById(`lane${laneNum}Speed`).textContent = `${metrics.avg_speed_kmh} km/h`;
}

function updateForecasts(forecasts) {
    if (forecasts["5_min"]) {
        document.getElementById("pred5Level").textContent = forecasts["5_min"].congestion_level;
        document.getElementById("pred5Pcu").textContent = forecasts["5_min"].predicted_pcu;
        document.getElementById("pred5Q").textContent = `${forecasts["5_min"].predicted_queue_m}m`;
    }
    if (forecasts["15_min"]) {
        document.getElementById("pred15Level").textContent = forecasts["15_min"].congestion_level;
        document.getElementById("pred15Pcu").textContent = forecasts["15_min"].predicted_pcu;
        document.getElementById("pred15Q").textContent = `${forecasts["15_min"].predicted_queue_m}m`;
    }
    if (forecasts["30_min"]) {
        document.getElementById("pred30Level").textContent = forecasts["30_min"].congestion_level;
        document.getElementById("pred30Pcu").textContent = forecasts["30_min"].predicted_pcu;
        document.getElementById("pred30Q").textContent = `${forecasts["30_min"].predicted_queue_m}m`;
    }
}

async function toggleGreenCorridor() {
    isGreenCorridor = !isGreenCorridor;
    const btn = document.getElementById("btnGreenCorridor");
    const pill = document.getElementById("hudCorridorPill");

    if (isGreenCorridor) {
        btn.textContent = "✅ CANCEL GREEN CORRIDOR";
        btn.classList.add("active");
        pill.style.display = "block";
        currentPhase = 1;
        countdownSec = 65;
    } else {
        btn.textContent = "🚨 TRIGGER GREEN CORRIDOR";
        btn.classList.remove("active");
        pill.style.display = "none";
    }

    try {
        await fetch(`/api/v1/incidents/emergency-override?enable=${isGreenCorridor}`, { method: "POST" });
    } catch (e) {}
}

async function fetchBenchmark() {
    try {
        const res = await fetch("/api/v1/simulation/benchmark");
        if (res.ok) {
            const data = await res.json();
            document.getElementById("delayRed").textContent = `-${data.delay_reduction_pct}%`;
        }
    } catch (e) {}
}

function changeCamera() {
    const sel = document.getElementById("camSelect").value;
    document.getElementById("cctvBadge").textContent = `AUTO-DETECT • ${sel.toUpperCase()} APPROACH`;
}

window.onload = init;

