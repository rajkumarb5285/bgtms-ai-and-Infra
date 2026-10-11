import React from 'react';
import type { VideoAnalysisResult } from '../types';
import { TrendingUp, Layers, Car, Clock, Zap, AlertTriangle, ShieldCheck } from 'lucide-react';

interface VideoResultsViewProps {
  result: VideoAnalysisResult;
}

export const VideoResultsView: React.FC<VideoResultsViewProps> = ({ result }) => {
  // Compute SVG chart coordinates for the timeline
  const maxPcu = Math.max(1, ...result.timeline.map((p) => p.pcu));
  const chartWidth = 600;
  const chartHeight = 160;

  const points = result.timeline.map((p, idx) => {
    const x = (idx / Math.max(1, result.timeline.length - 1)) * chartWidth;
    const y = chartHeight - (p.pcu / maxPcu) * (chartHeight - 30) - 10;
    return `${x},${y}`;
  });

  const pathD = points.length > 0 ? `M 0,${chartHeight} L ${points.join(' L ')} L ${chartWidth},${chartHeight} Z` : '';
  const lineD = points.length > 0 ? `M ${points.join(' L ')}` : '';

  return (
    <div className="results-container">
      {/* Top Hero Stats Bar */}
      <div className="stats-grid">
        <div className="stat-card glass-panel">
          <div className="stat-icon-wrapper icon-blue">
            <Car size={22} />
          </div>
          <div className="stat-info">
            <span className="stat-label">UNIQUE VEHICLES</span>
            <div className="stat-value font-mono">{result.total_unique_vehicles}</div>
            <span className="stat-sub">Tracked via ByteTrack MOT</span>
          </div>
        </div>

        <div className="stat-card glass-panel">
          <div className="stat-icon-wrapper icon-emerald">
            <Layers size={22} />
          </div>
          <div className="stat-info">
            <span className="stat-label">CUMULATIVE PCU</span>
            <div className="stat-value font-mono">{result.total_unique_pcu.toFixed(1)}</div>
            <span className="stat-sub">Total flow weight in clip</span>
          </div>
        </div>

        <div className="stat-card glass-panel">
          <div className="stat-icon-wrapper icon-amber">
            <TrendingUp size={22} />
          </div>
          <div className="stat-info">
            <span className="stat-label">PEAK DEMAND PCU</span>
            <div className="stat-value font-mono">{result.peak_pcu.toFixed(1)}</div>
            <span className="stat-sub">Max concurrent intersection load</span>
          </div>
        </div>

        <div className="stat-card glass-panel">
          <div className="stat-icon-wrapper icon-purple">
            <Clock size={22} />
          </div>
          <div className="stat-info">
            <span className="stat-label">VIDEO DURATION</span>
            <div className="stat-value font-mono">
              {result.duration_sec.toFixed(1)} <span className="unit">sec</span>
            </div>
            <span className="stat-sub">{result.processed_frames} sampled frames</span>
          </div>
        </div>

        <div className="stat-card glass-panel">
          <div className="stat-icon-wrapper icon-cyan">
            <Zap size={22} />
          </div>
          <div className="stat-info">
            <span className="stat-label">PROCESSING TIME</span>
            <div className="stat-value font-mono">
              {(result.latency_ms / 1000).toFixed(1)} <span className="unit">sec</span>
            </div>
            <span className="stat-sub">
              {(result.processed_frames / Math.max(0.1, result.latency_ms / 1000)).toFixed(1)} FPS throughput
            </span>
          </div>
        </div>
      </div>

      {/* Main Video Analysis Grid */}
      <div className="analysis-layout-grid">
        {/* Left Side: Peak Snapshot and Flow Timeline */}
        <div className="viewer-panel glass-panel">
          <div className="viewer-header">
            <div className="viewer-title-row">
              <span className="panel-dot"></span>
              <h3>Peak Traffic Snapshot ({result.peak_pcu.toFixed(1)} PCU Peak)</h3>
            </div>
            <span className="badge-pill badge-primary font-mono">{result.filename || 'video.mp4'}</span>
          </div>

          <div className="image-viewport">
            {result.peak_snapshot ? (
              <img src={result.peak_snapshot} alt="Peak Traffic Frame" className="viewport-image" />
            ) : (
              <div className="no-image-placeholder">No Snapshot Available</div>
            )}
          </div>

          {/* Traffic Flow Timeline Graph */}
          <div className="timeline-graph-card">
            <div className="graph-header">
              <h4>Traffic PCU Flow Dynamic Timeline</h4>
              <span className="font-mono text-muted">Time (seconds) vs PCU Load</span>
            </div>

            <div className="svg-chart-container">
              <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="timeline-svg">
                <defs>
                  <linearGradient id="pcuGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.45" />
                    <stop offset="100%" stopColor="#38bdf8" stopOpacity="0.0" />
                  </linearGradient>
                </defs>

                {/* Grid Lines */}
                <line x1="0" y1="40" x2={chartWidth} y2="40" stroke="rgba(255,255,255,0.06)" strokeDasharray="4" />
                <line x1="0" y1="80" x2={chartWidth} y2="80" stroke="rgba(255,255,255,0.06)" strokeDasharray="4" />
                <line x1="0" y1="120" x2={chartWidth} y2="120" stroke="rgba(255,255,255,0.06)" strokeDasharray="4" />

                {/* Area Fill */}
                {pathD && <path d={pathD} fill="url(#pcuGrad)" />}
                {/* Stroke Line */}
                {lineD && <path d={lineD} fill="none" stroke="#38bdf8" strokeWidth="3" />}
              </svg>

              <div className="timeline-axis-row font-mono">
                <span>0.0s</span>
                <span>{(result.duration_sec / 2).toFixed(1)}s</span>
                <span>{result.duration_sec.toFixed(1)}s</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Side: Unique Category Distribution & Video Incidents */}
        <div className="details-panel-column">
          {/* Class Breakdown */}
          <div className="breakdown-card glass-panel">
            <div className="card-title-row">
              <h4>Tracked Vehicle Categories</h4>
              <span className="count-pill">{result.total_unique_vehicles} Total</span>
            </div>

            <div className="classes-list">
              {Object.entries(result.class_breakdown).length === 0 ? (
                <p className="empty-text">No vehicles tracked.</p>
              ) : (
                Object.entries(result.class_breakdown).map(([cls, count]) => {
                  const pct = Math.round((count / Math.max(1, result.total_unique_vehicles)) * 100);
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

          {/* Incidents during video */}
          <div className="incidents-card glass-panel">
            <div className="card-title-row">
              <div className="flex-center-gap">
                <AlertTriangle size={18} className="icon-amber" />
                <h4>Video Incidents & Temporal Log</h4>
              </div>
              <span className="badge-count">
                {result.incidents.length > 0 ? `${result.incidents.length} EVENTS` : 'CLEAN'}
              </span>
            </div>

            <div className="incidents-feed">
              {result.incidents.length > 0 ? (
                result.incidents.map((inc, i) => (
                  <div key={i} className={`incident-alert-item alert-${inc.severity.toLowerCase()}`}>
                    <div className="alert-header">
                      <span className="alert-type">
                        {inc.video_time_sec !== undefined && (
                          <span className="font-mono text-cyan">[{inc.video_time_sec}s] </span>
                        )}
                        {inc.incident_type.replace(/_/g, ' ')}
                      </span>
                      <span className="alert-badge">{inc.severity}</span>
                    </div>
                    <p className="alert-desc">{inc.description}</p>
                  </div>
                ))
              ) : (
                <div className="no-incident-box">
                  <ShieldCheck size={24} className="icon-emerald" />
                  <div>
                    <strong>Zero Incidents Detected</strong>
                    <p>Smooth, unobstructed traffic flow across the video duration.</p>
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
