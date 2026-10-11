import React, { useState, useRef, useEffect } from 'react';
import type { VideoAnalysisResult } from '../types';
import {
  TrendingUp,
  Layers,
  Car,
  Clock,
  Zap,
  AlertTriangle,
  ShieldCheck,
  Play,
  Pause,
  RotateCcw,
  Volume2,
  VolumeX,
  Maximize2,
  SkipForward,
  SkipBack,
  Target,
  Film,
  Download
} from 'lucide-react';

interface VideoResultsViewProps {
  result: VideoAnalysisResult;
  videoSrc?: string | null;
}

export const VideoResultsView: React.FC<VideoResultsViewProps> = ({ result, videoSrc }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Fallback to latest video endpoint if local blob URL is missing
  const effectiveSrc = videoSrc || '/api/v1/video/latest';

  const [viewMode, setViewMode] = useState<'video' | 'snapshot'>('video');
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(result.duration_sec || 0);
  const [isMuted, setIsMuted] = useState<boolean>(true);
  const [playbackRate, setPlaybackRate] = useState<number>(1.0);
  const [videoError, setVideoError] = useState<boolean>(false);

  // Compute peak congestion point
  const peakPoint = result.timeline && result.timeline.length > 0
    ? result.timeline.reduce((max, p) => p.pcu > max.pcu ? p : max, result.timeline[0])
    : null;
  const peakSecond = peakPoint ? peakPoint.second : 0;

  // Format MM:SS helper
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Video event listeners
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleTimeUpdate = () => {
      setCurrentTime(video.currentTime);
    };

    const handleLoadedMetadata = () => {
      if (video.duration && !isNaN(video.duration)) {
        setDuration(video.duration);
      }
    };

    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);
    const handleEnded = () => setIsPlaying(false);

    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('loadedmetadata', handleLoadedMetadata);
    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);
    video.addEventListener('ended', handleEnded);

    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
      video.removeEventListener('ended', handleEnded);
    };
  }, [viewMode, effectiveSrc]);

  // Handle Play/Pause
  const togglePlay = () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
    } else {
      videoRef.current.play().catch(() => {});
    }
  };

  // Seek video to specific second
  const seekTo = (seconds: number) => {
    if (!videoRef.current) return;
    const maxDur = duration || result.duration_sec || 1;
    const clamped = Math.max(0, Math.min(seconds, maxDur));
    videoRef.current.currentTime = clamped;
    setCurrentTime(clamped);
  };

  const skipDelta = (deltaSec: number) => {
    if (!videoRef.current) return;
    seekTo(videoRef.current.currentTime + deltaSec);
  };

  const jumpToPeak = () => {
    setViewMode('video');
    seekTo(peakSecond);
  };

  const changePlaybackRate = (rate: number) => {
    if (!videoRef.current) return;
    videoRef.current.playbackRate = rate;
    setPlaybackRate(rate);
  };

  const toggleMute = () => {
    if (!videoRef.current) return;
    videoRef.current.muted = !isMuted;
    setIsMuted(!isMuted);
  };

  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen().catch(() => {});
    }
  };

  // Interactive timeline seek on click
  const handleTimelineClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, clickX / rect.width));
    const targetSec = ratio * (duration || result.duration_sec || 1);
    setViewMode('video');
    seekTo(targetSec);
  };

  const downloadSnapshot = () => {
    if (!result.peak_snapshot) return;
    const a = document.createElement('a');
    a.href = result.peak_snapshot;
    a.download = `bgtms_peak_${result.filename || 'traffic'}.jpg`;
    a.click();
  };

  // SVG chart coordinates for timeline
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

  // Scrub needle position on timeline SVG
  const totalVideoDur = Math.max(0.1, duration || result.duration_sec || 1);
  const scrubberRatio = Math.max(0, Math.min(1, currentTime / totalVideoDur));
  const scrubberX = scrubberRatio * chartWidth;

  const peakRatio = Math.max(0, Math.min(1, peakSecond / totalVideoDur));
  const peakX = peakRatio * chartWidth;

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
        {/* Left Side: Video Player / Peak Snapshot Viewer */}
        <div className="viewer-panel glass-panel">
          <div className="viewer-header">
            <div className="viewer-title-row">
              <span className="panel-dot"></span>
              {/* Tab Switcher: Video Player vs Peak Snapshot */}
              <div className="tab-switcher">
                <button
                  className={`tab-switch-btn ${viewMode === 'video' ? 'active' : ''}`}
                  onClick={() => setViewMode('video')}
                  title="Play/Pause full uploaded traffic video"
                >
                  <Film size={15} />
                  <span>Interactive Video Player</span>
                </button>
                <button
                  className={`tab-switch-btn ${viewMode === 'snapshot' ? 'active' : ''}`}
                  onClick={() => setViewMode('snapshot')}
                  title="View AI detected bounding boxes at maximum congestion"
                >
                  <Target size={15} />
                  <span>AI Peak Snapshot ({result.peak_pcu.toFixed(1)} PCU)</span>
                </button>
              </div>
            </div>

            <div className="viewer-controls">
              <button
                className="jump-peak-btn"
                onClick={jumpToPeak}
                title="Jump video player to peak congestion moment"
              >
                <Target size={14} />
                <span>Jump to Peak ({peakSecond.toFixed(1)}s)</span>
              </button>
              <span className="badge-pill badge-primary font-mono">{result.filename || 'video.mp4'}</span>
            </div>
          </div>

          {/* VIEWPORT: Video Player or Peak Snapshot */}
          <div className="image-viewport" ref={containerRef}>
            {viewMode === 'video' && !videoError ? (
              <div className="video-player-container">
                <video
                  ref={videoRef}
                  src={effectiveSrc}
                  className="video-element"
                  playsInline
                  muted={isMuted}
                  onClick={togglePlay}
                  onError={() => {
                    setVideoError(true);
                    setViewMode('snapshot');
                  }}
                />

                {/* Big Center Play Icon Overlay when paused */}
                {!isPlaying && (
                  <div className="video-center-overlay pulse" onClick={togglePlay}>
                    <Play size={28} fill="#38bdf8" />
                  </div>
                )}

                {/* Interactive Glassmorphic Video Controls Bar */}
                <div className="video-controls-overlay">
                  {/* Scrubber Progress Bar */}
                  <div
                    className="video-scrubber-track"
                    onClick={(e) => {
                      const rect = e.currentTarget.getBoundingClientRect();
                      const clickRatio = (e.clientX - rect.left) / rect.width;
                      seekTo(clickRatio * totalVideoDur);
                    }}
                  >
                    <div
                      className="video-scrubber-fill"
                      style={{ width: `${scrubberRatio * 100}%` }}
                    >
                      <div className="video-scrubber-thumb" />
                    </div>
                  </div>

                  {/* Buttons Row */}
                  <div className="video-controls-bar">
                    <div className="video-controls-group">
                      {/* Play / Pause Toggle Button */}
                      <button
                        className={`video-ctrl-btn ${isPlaying ? 'active' : ''}`}
                        onClick={togglePlay}
                        title={isPlaying ? 'Pause Video (Space)' : 'Play Video (Space)'}
                      >
                        {isPlaying ? <Pause size={18} /> : <Play size={18} fill="currentColor" />}
                      </button>

                      {/* Replay */}
                      <button
                        className="video-ctrl-btn"
                        onClick={() => seekTo(0)}
                        title="Replay from start"
                      >
                        <RotateCcw size={16} />
                      </button>

                      {/* Skip -5s */}
                      <button
                        className="video-ctrl-btn"
                        onClick={() => skipDelta(-5)}
                        title="Back 5 seconds"
                      >
                        <SkipBack size={16} />
                      </button>

                      {/* Skip +5s */}
                      <button
                        className="video-ctrl-btn"
                        onClick={() => skipDelta(5)}
                        title="Forward 5 seconds"
                      >
                        <SkipForward size={16} />
                      </button>

                      {/* Time display */}
                      <span className="video-time-chip font-mono">
                        {formatTime(currentTime)} / {formatTime(totalVideoDur)}
                      </span>

                      {/* Status indicator pill */}
                      <span className={`playback-status-pill ${!isPlaying ? 'paused' : ''}`}>
                        {isPlaying ? '▶ PLAYING' : '⏸ PAUSED'}
                      </span>
                    </div>

                    <div className="video-controls-group">
                      {/* Playback speed selector */}
                      {[0.5, 1.0, 1.5, 2.0].map((rate) => (
                        <button
                          key={rate}
                          className={`speed-badge ${playbackRate === rate ? 'active' : ''}`}
                          onClick={() => changePlaybackRate(rate)}
                        >
                          {rate}x
                        </button>
                      ))}

                      {/* Mute Toggle */}
                      <button
                        className="video-ctrl-btn"
                        onClick={toggleMute}
                        title={isMuted ? 'Unmute' : 'Mute'}
                      >
                        {isMuted ? <VolumeX size={16} /> : <Volume2 size={16} />}
                      </button>

                      {/* Fullscreen */}
                      <button
                        className="video-ctrl-btn"
                        onClick={toggleFullscreen}
                        title="Toggle Fullscreen"
                      >
                        <Maximize2 size={16} />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              /* Peak AI Snapshot View */
              <div style={{ position: 'relative', width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                {result.peak_snapshot ? (
                  <img
                    src={result.peak_snapshot}
                    alt="Peak Traffic Detection Frame"
                    className="viewport-image"
                  />
                ) : (
                  <div className="no-image-placeholder">No Snapshot Available</div>
                )}
                {result.peak_snapshot && (
                  <button
                    className="video-ctrl-btn"
                    onClick={downloadSnapshot}
                    style={{ position: 'absolute', top: 12, right: 12, background: 'rgba(15, 23, 42, 0.85)' }}
                    title="Download Annotated Peak Frame"
                  >
                    <Download size={15} style={{ marginRight: 6 }} />
                    <span>Download Image</span>
                  </button>
                )}
              </div>
            )}
          </div>

          {/* Traffic Flow Timeline Graph */}
          <div className="timeline-graph-card">
            <div className="graph-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <h4>Traffic PCU Flow Dynamic Timeline</h4>
                <span className="badge-pill badge-primary font-mono" style={{ fontSize: 10 }}>
                  Click anywhere to seek video
                </span>
              </div>
              <span className="font-mono text-muted">
                {currentTime.toFixed(1)}s • Time vs PCU Load
              </span>
            </div>

            <div className="svg-chart-container" title="Click anywhere to jump the video to that moment">
              <svg
                viewBox={`0 0 ${chartWidth} ${chartHeight}`}
                className="timeline-svg"
                onClick={handleTimelineClick}
              >
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

                {/* Peak Congestion Vertical Marker (Amber) */}
                {peakPoint && (
                  <g>
                    <line
                      x1={peakX}
                      y1="0"
                      x2={peakX}
                      y2={chartHeight}
                      stroke="#f59e0b"
                      strokeWidth="1.5"
                      strokeDasharray="3 3"
                    />
                    <circle cx={peakX} cy="15" r="4" fill="#f59e0b" />
                    <text
                      x={Math.min(chartWidth - 55, peakX + 6)}
                      y="19"
                      fill="#f59e0b"
                      fontSize="9"
                      fontFamily="monospace"
                      fontWeight="bold"
                    >
                      PEAK {result.peak_pcu.toFixed(1)}
                    </text>
                  </g>
                )}

                {/* Live Playback Scrubber Needle (Glowing Sky Blue) */}
                <g>
                  <line
                    x1={scrubberX}
                    y1="0"
                    x2={scrubberX}
                    y2={chartHeight}
                    stroke="#38bdf8"
                    strokeWidth="2.5"
                  />
                  <circle cx={scrubberX} cy={chartHeight - 10} r="5" fill="#38bdf8" stroke="#ffffff" strokeWidth="1.5" />
                </g>
              </svg>

              <div className="timeline-axis-row font-mono">
                <span>0.0s</span>
                <span style={{ color: '#f59e0b' }}>Peak @ {peakSecond.toFixed(1)}s</span>
                <span>{(totalVideoDur / 2).toFixed(1)}s</span>
                <span>{totalVideoDur.toFixed(1)}s</span>
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
                  <div
                    key={i}
                    className={`incident-alert-item alert-${inc.severity.toLowerCase()} interactive`}
                    onClick={() => {
                      if (inc.video_time_sec !== undefined) {
                        setViewMode('video');
                        seekTo(inc.video_time_sec);
                        if (videoRef.current) {
                          videoRef.current.play().catch(() => {});
                        }
                      }
                    }}
                    title={inc.video_time_sec !== undefined ? `Click to jump video to ${inc.video_time_sec}s` : undefined}
                  >
                    <div className="alert-header">
                      <span className="alert-type">
                        {inc.video_time_sec !== undefined && (
                          <span className="font-mono text-cyan" style={{ textDecoration: 'underline' }}>
                            [{inc.video_time_sec}s] ▶{' '}
                          </span>
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
