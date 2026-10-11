import React, { useEffect, useRef, useState } from 'react';
import type { LiveTelemetryPayload } from '../types';
import { Radio, AlertTriangle, Layers, Activity, Gauge } from 'lucide-react';

interface LiveStreamViewProps {
  isGreenCorridor: boolean;
  onToggleGreenCorridor: () => void;
}

export const LiveStreamView: React.FC<LiveStreamViewProps> = ({
  isGreenCorridor,
  onToggleGreenCorridor,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [telemetry, setTelemetry] = useState<LiveTelemetryPayload | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [countdown, setCountdown] = useState(38);
  const [currentPhase, setCurrentPhase] = useState(1);
  const [alerts, setAlerts] = useState<any[]>([]);

  // Simulation vehicles on canvas
  const vehiclesRef = useRef([
    { id: 1, type: 'auto_rickshaw', x: 280, y: 350, speed: 28, lane: 1, w: 42, h: 60, color: '#f59e0b', emergency: false },
    { id: 2, type: 'motorcycle', x: 340, y: 220, speed: 38, lane: 1, w: 24, h: 48, color: '#10b981', emergency: false },
    { id: 3, type: 'car', x: 620, y: 480, speed: 24, lane: 2, w: 56, h: 90, color: '#38bdf8', emergency: false },
    { id: 4, type: 'bus', x: 920, y: 300, speed: 18, lane: 3, w: 75, h: 160, color: '#a78bfa', emergency: false },
    { id: 5, type: 'motorcycle', x: 580, y: 320, speed: 32, lane: 2, w: 24, h: 48, color: '#10b981', emergency: false },
    { id: 6, type: 'auto_rickshaw', x: 670, y: 200, speed: 26, lane: 2, w: 42, h: 60, color: '#f59e0b', emergency: false },
    { id: 7, type: 'car', x: 930, y: 520, speed: 20, lane: 3, w: 56, h: 90, color: '#38bdf8', emergency: false },
    { id: 8, type: 'ambulance', x: 610, y: 120, speed: 45, lane: 2, w: 60, h: 105, color: '#ef4444', emergency: true },
  ]);

  // WebSocket connect
  useEffect(() => {
    let ws: WebSocket | null = null;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.hostname || '127.0.0.1'}:8000/ws/live-telemetry`;

    try {
      ws = new WebSocket(wsUrl);
      ws.onopen = () => setWsConnected(true);
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setTelemetry(data);
          if (data.new_incidents && data.new_incidents.length > 0) {
            setAlerts((prev) => [...data.new_incidents, ...prev].slice(0, 10));
          }
        } catch (e) {}
      };
      ws.onclose = () => setWsConnected(false);
      ws.onerror = () => setWsConnected(false);
    } catch (e) {
      setWsConnected(false);
    }

    return () => {
      ws?.close();
    };
  }, []);

  // Countdown timer
  useEffect(() => {
    const timer = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          setCurrentPhase((p) => (p === 1 ? 2 : 1));
          return currentPhase === 1 ? 34 : 52;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [currentPhase]);

  // Canvas render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;

    const render = () => {
      const w = canvas.width;
      const h = canvas.height;

      // Dark asphalt
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, w, h);

      // Trapezoid Road Perspective
      ctx.fillStyle = '#1e293b';
      ctx.beginPath();
      ctx.moveTo(w * 0.15, h);
      ctx.lineTo(w * 0.35, h * 0.15);
      ctx.lineTo(w * 0.75, h * 0.15);
      ctx.lineTo(w * 0.95, h);
      ctx.closePath();
      ctx.fill();

      // Lane dividers
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
      ctx.lineWidth = 3;
      ctx.setLineDash([20, 15]);

      ctx.beginPath();
      ctx.moveTo(w * 0.42, h);
      ctx.lineTo(w * 0.48, h * 0.15);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(w * 0.68, h);
      ctx.lineTo(w * 0.62, h * 0.15);
      ctx.stroke();
      ctx.setLineDash([]);

      // Stop Line
      const isGreen = currentPhase === 1 || isGreenCorridor;
      ctx.strokeStyle = isGreen ? '#10b981' : '#ef4444';
      ctx.lineWidth = 6;
      ctx.beginPath();
      ctx.moveTo(w * 0.18, h * 0.85);
      ctx.lineTo(w * 0.92, h * 0.85);
      ctx.stroke();

      ctx.fillStyle = isGreen ? '#10b981' : '#ef4444';
      ctx.font = 'bold 14px JetBrains Mono';
      ctx.fillText(
        `STOP LINE [${isGreen ? 'SIGNAL GREEN' : 'SIGNAL RED'}]`,
        w * 0.2,
        h * 0.83
      );

      // Move & draw vehicles
      vehiclesRef.current.forEach((v) => {
        let step = v.speed / 18.0;
        if (!isGreen && v.y > h * 0.68 && v.y < h * 0.85) {
          step = 0.2; // stopped
        }
        v.y += step;
        if (v.y > h + 100) {
          v.y = h * 0.15;
        }

        // Box
        ctx.strokeStyle = v.emergency ? '#ef4444' : v.color;
        ctx.lineWidth = v.emergency ? 3 : 2;
        ctx.strokeRect(v.x - v.w / 2, v.y - v.h / 2, v.w, v.h);

        ctx.fillStyle = v.emergency ? 'rgba(239, 68, 68, 0.25)' : 'rgba(15, 23, 42, 0.5)';
        ctx.fillRect(v.x - v.w / 2, v.y - v.h / 2, v.w, v.h);

        // Flashing beacon for emergency
        if (v.emergency) {
          ctx.fillStyle = Date.now() % 400 < 200 ? '#ef4444' : '#38bdf8';
          ctx.beginPath();
          ctx.arc(v.x, v.y - v.h / 2 - 8, 6, 0, Math.PI * 2);
          ctx.fill();
        }

        // Tag
        ctx.fillStyle = v.emergency ? '#ef4444' : v.color;
        const tag = `#${v.id} ${v.type.toUpperCase()} ${v.speed}km/h`;
        ctx.font = '11px JetBrains Mono';
        const tagW = ctx.measureText(tag).width;
        ctx.fillRect(v.x - v.w / 2, v.y - v.h / 2 - 16, tagW + 8, 16);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(tag, v.x - v.w / 2 + 4, v.y - v.h / 2 - 4);
      });

      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animId);
  }, [currentPhase, isGreenCorridor]);

  const simulateIncident = async (type: string) => {
    try {
      const res = await fetch(`/api/v1/incidents/simulate?incident_type=${type}`, {
        method: 'POST',
      });
      if (res.ok) {
        const data = await res.json();
        if (data.incident) {
          setAlerts((prev) => [data.incident, ...prev].slice(0, 10));
        }
      }
    } catch (e) {}
  };

  return (
    <div className="live-stream-container">
      {/* Top Telemetry Stats */}
      <div className="stats-grid">
        <div className="stat-card glass-panel">
          <div className="stat-icon-wrapper icon-blue">
            <Radio size={22} />
          </div>
          <div className="stat-info">
            <span className="stat-label">STREAM LINK</span>
            <div className="stat-value font-mono">
              <span className={`status-badge ${wsConnected ? 'badge-green' : 'badge-amber'}`}>
                {wsConnected ? '10 Hz WEBSOCKET' : 'CONNECTING...'}
              </span>
            </div>
            <span className="stat-sub">Central Silk Board Cam 01</span>
          </div>
        </div>

        <div className="stat-card glass-panel">
          <div className="stat-icon-wrapper icon-emerald">
            <Layers size={22} />
          </div>
          <div className="stat-info">
            <span className="stat-label">APPROACH LOAD</span>
            <div className="stat-value font-mono">
              {telemetry?.approach_state?.total_pcu?.toFixed(1) || '24.5'}{' '}
              <span className="unit">PCU</span>
            </div>
            <span className="stat-sub">Hosur Road Inbound Carriageway</span>
          </div>
        </div>

        <div className="stat-card glass-panel">
          <div className="stat-icon-wrapper icon-amber">
            <Activity size={22} />
          </div>
          <div className="stat-info">
            <span className="stat-label">ADAPTIVE CYCLE</span>
            <div className="stat-value font-mono">
              {countdown} <span className="unit">sec left</span>
            </div>
            <span className="stat-sub">
              {currentPhase === 1 ? 'Phase 1: North-South Green' : 'Phase 2: East-West Green'}
            </span>
          </div>
        </div>

        <div className="stat-card glass-panel">
          <div className="stat-icon-wrapper icon-purple">
            <Gauge size={22} />
          </div>
          <div className="stat-info">
            <span className="stat-label">STREAM LATENCY</span>
            <div className="stat-value font-mono">
              {telemetry?.latency_ms?.toFixed(1) || '18.4'} <span className="unit">ms</span>
            </div>
            <span className="stat-sub">
              {telemetry?.fps?.toFixed(0) || '54'} FPS Realtime
            </span>
          </div>
        </div>
      </div>

      {/* Main Live Canvas + Control Sidebar */}
      <div className="analysis-layout-grid">
        {/* Canvas Viewport */}
        <div className="viewer-panel glass-panel">
          <div className="viewer-header">
            <div className="viewer-title-row">
              <span className="live-pulse"></span>
              <h3>Live Intersection Optical Telemetry (1280x720)</h3>
            </div>
            <div className="live-badge-group">
              <span className="badge-model font-mono">SILK BOARD • CAM-01</span>
              <button
                className={`badge-pill ${isGreenCorridor ? 'badge-danger animate-pulse' : 'badge-primary'}`}
                style={{ cursor: 'pointer', border: 'none' }}
                onClick={onToggleGreenCorridor}
              >
                {isGreenCorridor ? '🚨 PREEMPTION ACTIVE (CANCEL)' : '⚡ TRIGGER PREEMPTION'}
              </button>
            </div>
          </div>

          <div className="image-viewport canvas-wrapper">
            <canvas ref={canvasRef} width={1280} height={720} className="viewport-image" />
          </div>
        </div>

        {/* Live Incident Dispatch & Simulation Tools */}
        <div className="details-panel-column">
          {/* Quick Simulation Panel */}
          <div className="breakdown-card glass-panel">
            <div className="card-title-row">
              <h4>Operator Simulation Injection</h4>
              <span className="tag-cyan font-mono">M2 TEST HARNESS</span>
            </div>
            <p className="sim-helper-text">
              Inject real-time kinematic collisions or ambulance emergencies into the live pipeline:
            </p>
            <div className="sim-btn-row">
              <button className="sim-btn sim-btn-red" onClick={() => simulateIncident('COLLISION')}>
                💥 Inject Collision
              </button>
              <button className="sim-btn sim-btn-blue" onClick={() => simulateIncident('EMERGENCY')}>
                🚑 Inject Ambulance
              </button>
              <button className="sim-btn sim-btn-amber" onClick={() => simulateIncident('WRONG_WAY')}>
                ⚠️ Inject Wrong-Way
              </button>
            </div>
          </div>

          {/* Realtime Incident Dispatch */}
          <div className="incidents-card glass-panel">
            <div className="card-title-row">
              <div className="flex-center-gap">
                <AlertTriangle size={18} className="icon-amber" />
                <h4>Live Dispatch Stream</h4>
              </div>
              <span className="badge-count">{alerts.length} ALERTS</span>
            </div>

            <div className="incidents-feed">
              {alerts.length === 0 ? (
                <div className="no-incident-box">
                  <span className="live-pulse"></span>
                  <div>
                    <strong>Normal Carriageway Operations</strong>
                    <p>Live AI vision continuously scanning for kinematic disruptions.</p>
                  </div>
                </div>
              ) : (
                alerts.map((al, idx) => (
                  <div key={idx} className={`incident-alert-item alert-${(al.severity || 'info').toLowerCase()}`}>
                    <div className="alert-header">
                      <span className="alert-type">{(al.incident_type || 'INCIDENT').replace(/_/g, ' ')}</span>
                      <span className="alert-badge">{al.severity || 'WARNING'}</span>
                    </div>
                    <p className="alert-desc">{al.description}</p>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
