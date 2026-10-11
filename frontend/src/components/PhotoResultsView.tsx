import React, { useState } from 'react';
import type { PhotoAnalysisResult } from '../types';
import {
  Download,
  Eye,
  Activity,
  Layers,
  Car,
  Gauge,
  AlertTriangle,
  Zap,
  CheckCircle2,
} from 'lucide-react';

interface PhotoResultsViewProps {
  result: PhotoAnalysisResult;
  originalImageSrc?: string;
}

export const PhotoResultsView: React.FC<PhotoResultsViewProps> = ({
  result,
  originalImageSrc,
}) => {
  const [showOriginal, setShowOriginal] = useState(false);

  const getCongestionClass = (level: string) => {
    switch (level) {
      case 'LOW':
        return 'badge-green';
      case 'MEDIUM':
        return 'badge-amber';
      case 'HIGH':
        return 'badge-orange';
      case 'CRITICAL':
        return 'badge-red';
      default:
        return 'badge-green';
    }
  };

  const downloadAnnotatedImage = () => {
    const a = document.createElement('a');
    a.href = result.annotated_image;
    a.download = `bgtms_annotated_${result.filename || 'traffic'}.jpg`;
    a.click();
  };

  const downloadJsonReport = () => {
    const blob = new Blob([JSON.stringify(result, null, 2)], {
      type: 'application/json',
    });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `bgtms_report_${result.filename || 'traffic'}.json`;
    a.click();
  };

  return (
    <div className="results-container">
      {/* Top Hero Stats Bar */}
      <div className="stats-grid">
        <div className="stat-card glass-panel">
          <div className="stat-icon-wrapper icon-blue">
            <Car size={22} />
          </div>
          <div className="stat-info">
            <span className="stat-label">VEHICLES DETECTED</span>
            <div className="stat-value font-mono">{result.detections_count}</div>
            <span className="stat-sub">Across approach carriageway</span>
          </div>
        </div>

        <div className="stat-card glass-panel">
          <div className="stat-icon-wrapper icon-emerald">
            <Layers size={22} />
          </div>
          <div className="stat-info">
            <span className="stat-label">TOTAL IRC PCU</span>
            <div className="stat-value font-mono">{result.total_pcu.toFixed(1)}</div>
            <span className="stat-sub">Indian Roads Congress standard</span>
          </div>
        </div>

        <div className="stat-card glass-panel">
          <div className="stat-icon-wrapper icon-amber">
            <Activity size={22} />
          </div>
          <div className="stat-info">
            <span className="stat-label">CONGESTION LEVEL</span>
            <div className="stat-badge-row">
              <span className={`status-badge ${getCongestionClass(result.congestion_level)}`}>
                {result.congestion_level}
              </span>
            </div>
            <span className="stat-sub">Silk Board junction approach</span>
          </div>
        </div>

        <div className="stat-card glass-panel">
          <div className="stat-icon-wrapper icon-purple">
            <Gauge size={22} />
          </div>
          <div className="stat-info">
            <span className="stat-label">SPEED PROXY</span>
            <div className="stat-value font-mono">
              {result.average_speed_kmh.toFixed(1)} <span className="unit">km/h</span>
            </div>
            <span className="stat-sub">Homography metric estimation</span>
          </div>
        </div>

        <div className="stat-card glass-panel">
          <div className="stat-icon-wrapper icon-cyan">
            <Zap size={22} />
          </div>
          <div className="stat-info">
            <span className="stat-label">PIPELINE LATENCY</span>
            <div className="stat-value font-mono">
              {result.latency_ms.toFixed(0)} <span className="unit">ms</span>
            </div>
            <span className="stat-sub">
              ~{(1000 / Math.max(1, result.latency_ms)).toFixed(0)} FPS throughput
            </span>
          </div>
        </div>
      </div>

      {/* Main Analysis Layout */}
      <div className="analysis-layout-grid">
        {/* Left Side: Annotated Image Viewer */}
        <div className="viewer-panel glass-panel">
          <div className="viewer-header">
            <div className="viewer-title-row">
              <span className="panel-dot"></span>
              <h3>Vision Detection Output ({result.width}x{result.height})</h3>
            </div>

            <div className="viewer-controls">
              {originalImageSrc && (
                <button
                  className={`toggle-view-btn ${showOriginal ? 'active' : ''}`}
                  onClick={() => setShowOriginal(!showOriginal)}
                >
                  <Eye size={15} />
                  <span>{showOriginal ? 'Show Annotated' : 'Show Original'}</span>
                </button>
              )}
              <button className="icon-action-btn" onClick={downloadAnnotatedImage} title="Download Annotated Image">
                <Download size={15} />
                <span>Export Image</span>
              </button>
              <button className="icon-action-btn" onClick={downloadJsonReport} title="Export JSON Report">
                <span>Export JSON</span>
              </button>
            </div>
          </div>

          <div className="image-viewport">
            <img
              src={showOriginal && originalImageSrc ? originalImageSrc : result.annotated_image}
              alt="Traffic Detection Result"
              className="viewport-image"
            />
          </div>

          <div className="viewport-footer">
            <div className="filename-tag font-mono">
              <span>SOURCE:</span> {result.filename || 'uploaded_image.jpg'}
            </div>
            <div className="resolution-tag font-mono">
              RESOLUTION: {result.width}x{result.height} px
            </div>
          </div>
        </div>

        {/* Right Side: Microscopic Breakdown & Incidents */}
        <div className="details-panel-column">
          {/* Vehicle Class Breakdown */}
          <div className="breakdown-card glass-panel">
            <div className="card-title-row">
              <h4>Class Distribution Breakdown</h4>
              <span className="count-pill">{result.detections_count} Total</span>
            </div>

            <div className="classes-list">
              {Object.entries(result.class_breakdown).length === 0 ? (
                <p className="empty-text">No vehicles detected above threshold.</p>
              ) : (
                Object.entries(result.class_breakdown).map(([cls, count]) => {
                  const pct = Math.round((count / Math.max(1, result.detections_count)) * 100);
                  return (
                    <div key={cls} className="class-item-row">
                      <div className="class-meta">
                        <span className="class-name">{cls.replace(/_/g, ' ').toUpperCase()}</span>
                        <strong className="class-count font-mono">{count} ({pct}%)</strong>
                      </div>
                      <div className="progress-bar-track">
                        <div
                          className={`progress-bar-fill fill-${cls}`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>

          {/* Microscopic Lane Telemetry */}
          {result.approach_state && result.approach_state.lanes && (
            <div className="lanes-card glass-panel">
              <div className="card-title-row">
                <h4>Lane Micro-Telemetry (Silk Board)</h4>
                <span className="tag-green font-mono">3 CARRIAGEWAYS</span>
              </div>

              <div className="lanes-stack">
                {result.approach_state.lanes.map((lane) => (
                  <div key={lane.lane_id} className="lane-row-card">
                    <div className="lane-header-row">
                      <span className="lane-id">LANE {lane.lane_id}</span>
                      <span className="lane-counts font-mono">
                        {lane.vehicle_count} veh • {lane.pcu_count.toFixed(1)} PCU
                      </span>
                    </div>
                    <div className="progress-bar-track">
                      <div
                        className="progress-bar-fill fill-lane"
                        style={{ width: `${Math.min(100, lane.occupancy_pct)}%` }}
                      />
                    </div>
                    <div className="lane-sub-metrics font-mono">
                      <span>Queue: <strong>{lane.queue_length_m.toFixed(0)}m</strong></span>
                      <span>Speed: <strong>{lane.avg_speed_kmh.toFixed(1)} km/h</strong></span>
                      <span>Occ: <strong>{lane.occupancy_pct.toFixed(0)}%</strong></span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Incidents Box */}
          <div className="incidents-card glass-panel">
            <div className="card-title-row">
              <div className="flex-center-gap">
                <AlertTriangle size={18} className="icon-amber" />
                <h4>Safety & Incident Analysis</h4>
              </div>
              <span className="badge-count">
                {result.incidents && result.incidents.length > 0
                  ? `${result.incidents.length} ALERTS`
                  : 'SAFE FLOW'}
              </span>
            </div>

            <div className="incidents-feed">
              {result.incidents && result.incidents.length > 0 ? (
                result.incidents.map((inc, i) => (
                  <div
                    key={inc.incident_id || i}
                    className={`incident-alert-item alert-${inc.severity.toLowerCase()}`}
                  >
                    <div className="alert-header">
                      <span className="alert-type">{inc.incident_type.replace(/_/g, ' ')}</span>
                      <span className="alert-badge">{inc.severity}</span>
                    </div>
                    <p className="alert-desc">{inc.description}</p>
                  </div>
                ))
              ) : (
                <div className="no-incident-box">
                  <CheckCircle2 size={24} className="icon-emerald" />
                  <div>
                    <strong>Normal Approach Kinematics</strong>
                    <p>No sudden decelerations, collisions, or lane blockages detected.</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
