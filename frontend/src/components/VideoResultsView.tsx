import React, { useState, useRef, useEffect, useCallback } from 'react';
import type { VideoAnalysisResult, VideoTimelinePoint, VideoBoxItem } from '../types';
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
  Download,
  Eye,
  EyeOff,
  Sparkles,
  RefreshCw
} from 'lucide-react';

interface VideoResultsViewProps {
  result: VideoAnalysisResult;
  videoSrc?: string | null;
  onRescan?: () => void;
}

const CLASS_COLORS: Record<string, { stroke: string; fill: string; icon: string }> = {
  car: { stroke: '#38bdf8', fill: 'rgba(56, 189, 248, 0.22)', icon: '🚗' },
  motorcycle: { stroke: '#10b981', fill: 'rgba(16, 185, 129, 0.22)', icon: '🛵' },
  auto_rickshaw: { stroke: '#f59e0b', fill: 'rgba(245, 158, 11, 0.22)', icon: '🛺' },
  bus: { stroke: '#a855f7', fill: 'rgba(168, 85, 247, 0.22)', icon: '🚌' },
  truck: { stroke: '#94a3b8', fill: 'rgba(148, 163, 184, 0.22)', icon: '🚛' },
  bicycle: { stroke: '#06b6d4', fill: 'rgba(6, 182, 212, 0.22)', icon: '🚲' },
  pedestrian: { stroke: '#ec4899', fill: 'rgba(236, 72, 153, 0.22)', icon: '🚶' },
  emergency_vehicle: { stroke: '#ef4444', fill: 'rgba(239, 68, 68, 0.35)', icon: '🚑' },
};

export const VideoResultsView: React.FC<VideoResultsViewProps> = ({
  result,
  videoSrc,
  onRescan
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
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
  const [showAiBoxes, setShowAiBoxes] = useState<boolean>(true);
  const [liveBoxesCount, setLiveBoxesCount] = useState<number>(0);
  const [isRescanning, setIsRescanning] = useState<boolean>(false);

  // Check if timeline contains pre-computed frame bounding boxes
  const hasBoxesInTimeline = Boolean(
    result.timeline && result.timeline.some((p) => p.boxes && p.boxes.length > 0)
  );

  // Compute peak congestion point
  const peakPoint = result.timeline && result.timeline.length > 0
    ? result.timeline.reduce((max, p) => p.pcu > max.pcu ? p : max, result.timeline[0])
    : null;
  const peakSecond = result.peak_second !== undefined
    ? result.peak_second
    : (peakPoint ? peakPoint.second : 0);

  // Format MM:SS helper
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Check if an emergency vehicle (ambulance) incident occurred in this video
  const emergencyIncident = result.incidents?.find(
    (inc) => inc.incident_type === 'EMERGENCY_VEHICLE_APPROACH' || inc.severity === 'CRITICAL' || inc.details?.class_name === 'emergency_vehicle'
  );

  // Find nearest timeline point for current playback second
  const findClosestPoint = useCallback((sec: number): VideoTimelinePoint | null => {
    if (!result.timeline || result.timeline.length === 0) return null;
    let closest = result.timeline[0];
    let minDiff = Math.abs(sec - closest.second);
    for (let i = 1; i < result.timeline.length; i++) {
      const diff = Math.abs(sec - result.timeline[i].second);
      if (diff < minDiff) {
        minDiff = diff;
        closest = result.timeline[i];
      }
    }
    return closest;
  }, [result.timeline]);

  // Draw real-time detection bounding boxes on video overlay canvas
  const drawBoxesOnCanvas = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || viewMode !== 'video') return;

    const rect = video.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;

    if (canvas.width !== Math.round(rect.width) || canvas.height !== Math.round(rect.height)) {
      canvas.width = Math.round(rect.width);
      canvas.height = Math.round(rect.height);
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!showAiBoxes) {
      setLiveBoxesCount(0);
      return;
    }

    const closest = findClosestPoint(video.currentTime);
    const boxes: VideoBoxItem[] = closest?.boxes || [];
    setLiveBoxesCount(boxes.length);

    boxes.forEach((box) => {
      const x = box.norm_x1 * canvas.width;
      const y = box.norm_y1 * canvas.height;
      const w = Math.max(14, (box.norm_x2 - box.norm_x1) * canvas.width);
      const h = Math.max(14, (box.norm_y2 - box.norm_y1) * canvas.height);

      const colorData = CLASS_COLORS[box.class_name] || {
        stroke: '#38bdf8',
        fill: 'rgba(56, 189, 248, 0.22)',
        icon: '🚘'
      };

      const isEmergency = box.class_name === 'emergency_vehicle';

      // Shaded box fill (Crimson pulse for emergency vehicle)
      ctx.fillStyle = isEmergency ? 'rgba(239, 68, 68, 0.40)' : colorData.fill;
      ctx.fillRect(x, y, w, h);

      // Neon glowing border
      ctx.strokeStyle = isEmergency ? '#ef4444' : colorData.stroke;
      ctx.lineWidth = isEmergency ? 4 : 2.5;
      ctx.shadowColor = isEmergency ? '#ef4444' : colorData.stroke;
      ctx.shadowBlur = isEmergency ? 16 : 6;
      ctx.strokeRect(x, y, w, h);
      ctx.shadowBlur = 0;

      // Label chip above bounding box
      const label = isEmergency
        ? `🚨 AMBULANCE #${box.track_id} PREEMPTION: STOP ALL`
        : `${colorData.icon} ${box.class_name.replace(/_/g, ' ').toUpperCase()} #${box.track_id} ${(box.confidence * 100).toFixed(0)}%`;
      ctx.font = isEmergency ? 'bold 12px Inter, system-ui, sans-serif' : 'bold 11px Inter, system-ui, sans-serif';
      const textWidth = ctx.measureText(label).width;
      const tagH = isEmergency ? 22 : 18;
      const tagW = textWidth + 14;
      const tagY = Math.max(0, y - tagH - 2);

      ctx.fillStyle = isEmergency ? 'rgba(185, 28, 28, 0.95)' : 'rgba(15, 23, 42, 0.88)';
      ctx.fillRect(x, tagY, tagW, tagH);
      ctx.strokeStyle = isEmergency ? '#fca5a5' : colorData.stroke;
      ctx.lineWidth = isEmergency ? 1.5 : 1;
      ctx.strokeRect(x, tagY, tagW, tagH);

      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, x + 7, tagY + (isEmergency ? 15 : 13));
    });
  }, [findClosestPoint, showAiBoxes, viewMode]);

  // Video event listeners
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleTimeUpdate = () => {
      setCurrentTime(video.currentTime);
      drawBoxesOnCanvas();
    };

    const handleLoadedMetadata = () => {
      if (video.duration && !isNaN(video.duration)) {
        setDuration(video.duration);
      }
      drawBoxesOnCanvas();
    };

    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => {
      setIsPlaying(false);
      drawBoxesOnCanvas();
    };
    const handleEnded = () => {
      setIsPlaying(false);
      drawBoxesOnCanvas();
    };

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
  }, [viewMode, effectiveSrc, drawBoxesOnCanvas]);

  // 60 FPS requestAnimationFrame loop during active video playback
  useEffect(() => {
    let animId: number;
    const loop = () => {
      drawBoxesOnCanvas();
      if (isPlaying) {
        animId = requestAnimationFrame(loop);
      }
    };

    if (isPlaying) {
      animId = requestAnimationFrame(loop);
    } else {
      drawBoxesOnCanvas();
    }

    return () => cancelAnimationFrame(animId);
  }, [isPlaying, drawBoxesOnCanvas]);

  // Window resize listener to keep canvas scaled
  useEffect(() => {
    const handleResize = () => drawBoxesOnCanvas();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [drawBoxesOnCanvas]);

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
    drawBoxesOnCanvas();
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

  const triggerRescan = async () => {
    if (onRescan) {
      setIsRescanning(true);
      try {
        await onRescan();
      } finally {
        setIsRescanning(false);
      }
    }
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
      {/* Emergency Vehicle Preemption Banner */}
      {emergencyIncident && (
        <div className="emergency-preemption-banner">
          <div className="emergency-banner-left">
            <span className="emergency-siren-badge">🚨 CRITICAL INCIDENT</span>
            <div className="emergency-banner-text">
              <strong>EMERGENCY VEHICLE (AMBULANCE) DETECTED — AUTOMATED GREEN CORRIDOR PREEMPTION ACTIVE</strong>
              <p>
                Inbound at {emergencyIncident.video_time_sec !== undefined ? `${emergencyIncident.video_time_sec}s` : 'active traffic'}.
                All conflicting approaches held at RED. STOP ALL CONFLICTING VEHICLES.
              </p>
            </div>
          </div>
          {emergencyIncident.video_time_sec !== undefined && (
            <button
              className="jump-emergency-btn"
              onClick={() => {
                setViewMode('video');
                seekTo(emergencyIncident.video_time_sec!);
                if (videoRef.current) {
                  videoRef.current.play().catch(() => {});
                }
              }}
              title="Jump directly to the ambulance pass in video"
            >
              <Play size={14} fill="currentColor" />
              <span>Jump to Ambulance ({emergencyIncident.video_time_sec}s)</span>
            </button>
          )}
        </div>
      )}

      {/* Rescan Banner Prompt if video was analyzed without live frame bounding boxes */}
      {!hasBoxesInTimeline && onRescan && (
        <div className="rescan-prompt-banner">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <Sparkles size={18} />
            <span>
              <strong>Enable Live Moving Bounding Boxes:</strong> This video was processed before real-time box streaming was activated. Click Re-Scan to overlay detection boxes on every vehicle during playback!
            </span>
          </div>
          <button className="rescan-btn" onClick={triggerRescan} disabled={isRescanning}>
            <RefreshCw size={14} className={isRescanning ? 'spin' : ''} />
            <span>{isRescanning ? 'Scanning Video...' : 'Scan Live Detections'}</span>
          </button>
        </div>
      )}

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
        {/* Left Side: Video Player with Real-Time Bounding Box Overlay */}
        <div className="viewer-panel glass-panel">
          <div className="viewer-header">
            <div className="viewer-title-row">
              <span className="panel-dot"></span>
              {/* Tab Switcher: Video Player vs Peak Snapshot */}
              <div className="tab-switcher">
                <button
                  className={`tab-switch-btn ${viewMode === 'video' ? 'active' : ''}`}
                  onClick={() => setViewMode('video')}
                  title="Play/Pause full uploaded traffic video with live bounding boxes"
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
              {/* AI Overlay toggle */}
              {viewMode === 'video' && (
                <button
                  className={`ai-overlay-badge ${showAiBoxes ? '' : 'inactive'}`}
                  onClick={() => setShowAiBoxes(!showAiBoxes)}
                  title="Toggle real-time detection bounding box overlay on video"
                >
                  {showAiBoxes ? <Eye size={14} /> : <EyeOff size={14} />}
                  <span>{showAiBoxes ? `Detections ON (${liveBoxesCount})` : 'Detections OFF'}</span>
                </button>
              )}

              {emergencyIncident?.video_time_sec !== undefined && (
                <button
                  className="jump-emergency-btn"
                  style={{ padding: '5px 11px', fontSize: 11 }}
                  onClick={() => {
                    setViewMode('video');
                    seekTo(emergencyIncident.video_time_sec!);
                  }}
                  title="Jump to Ambulance Moment"
                >
                  <Play size={12} fill="currentColor" />
                  <span>Ambulance ({emergencyIncident.video_time_sec}s)</span>
                </button>
              )}

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

          {/* VIEWPORT: Video Player with Real-Time Canvas Overlay */}
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

                {/* Real-time Bounding Box Overlay Canvas */}
                <canvas
                  ref={canvasRef}
                  className="video-canvas-overlay"
                  onClick={togglePlay}
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

                      {/* Status indicator pill with live vehicle count */}
                      <span className={`playback-status-pill ${!isPlaying ? 'paused' : ''}`}>
                        {isPlaying ? `▶ PLAYING (${liveBoxesCount} detected)` : '⏸ PAUSED'}
                      </span>
                    </div>

                    <div className="video-controls-group">
                      {/* Bounding box toggle button */}
                      <button
                        className={`video-ctrl-btn ${showAiBoxes ? 'active' : ''}`}
                        onClick={() => setShowAiBoxes(!showAiBoxes)}
                        title="Toggle live bounding box overlay"
                      >
                        {showAiBoxes ? <Eye size={16} /> : <EyeOff size={16} />}
                      </button>

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
                {currentTime.toFixed(1)}s • Time vs PCU Load ({liveBoxesCount} vehicles)
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
