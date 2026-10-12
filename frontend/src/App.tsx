import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { UploadDropzone } from './components/UploadDropzone';
import { PhotoResultsView } from './components/PhotoResultsView';
import { VideoResultsView } from './components/VideoResultsView';
import { LiveStreamView } from './components/LiveStreamView';
import type { PhotoAnalysisResult, VideoAnalysisResult } from './types';
import './App.css';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'photo' | 'video' | 'live'>('photo');
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.35);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [photoResult, setPhotoResult] = useState<PhotoAnalysisResult | null>(null);
  const [originalImageSrc, setOriginalImageSrc] = useState<string | undefined>(undefined);
  const [videoResult, setVideoResult] = useState<VideoAnalysisResult | null>(null);
  const [videoSrc, setVideoSrc] = useState<string | null>(null);

  const [isGreenCorridor, setIsGreenCorridor] = useState<boolean>(false);
  const [serverStatus, setServerStatus] = useState<'connected' | 'offline' | 'checking'>('checking');

  // Check health on mount with fast 4s polling and instant manual retry
  const checkServer = async () => {
    try {
      const res = await fetch('/api/v1/health');
      if (res.ok) {
        setServerStatus('connected');
      } else {
        setServerStatus('offline');
      }
    } catch (e) {
      setServerStatus('offline');
    }
  };

  useEffect(() => {
    checkServer();
    const interval = setInterval(checkServer, 4000);
    return () => clearInterval(interval);
  }, []);

  // Handle Photo Upload
  const handlePhotoUpload = async (file: File) => {
    setIsLoading(true);
    setErrorMessage(null);

    // Save local preview URL for original view
    const localUrl = URL.createObjectURL(file);
    setOriginalImageSrc(localUrl);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch(
        `/api/v1/analyze/photo?confidence_threshold=${confidenceThreshold}`,
        {
          method: 'POST',
          body: formData,
        }
      );

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Analysis failed' }));
        throw new Error(err.detail || 'Photo analysis failed');
      }

      const data: PhotoAnalysisResult = await res.json();
      setPhotoResult(data);
    } catch (e: any) {
      setErrorMessage(e.message || 'An error occurred during photo detection.');
    } finally {
      setIsLoading(false);
    }
  };

  const [lastVideoFile, setLastVideoFile] = useState<File | null>(null);

  // Handle Video Upload
  const handleVideoUpload = async (file: File) => {
    setIsLoading(true);
    setErrorMessage(null);
    setLastVideoFile(file);

    // Track local video URL for instantaneous playback with play/pause
    if (videoSrc && videoSrc.startsWith('blob:')) {
      URL.revokeObjectURL(videoSrc);
    }
    const localUrl = URL.createObjectURL(file);
    setVideoSrc(localUrl);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch(
        `/api/v1/analyze/video?confidence_threshold=${confidenceThreshold}&max_seconds=360`,
        {
          method: 'POST',
          body: formData,
        }
      );

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Analysis failed' }));
        throw new Error(err.detail || 'Video analysis failed');
      }

      const data: VideoAnalysisResult = await res.json();
      setVideoResult(data);
    } catch (e: any) {
      setErrorMessage(e.message || 'An error occurred during video analysis.');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle Sample Photo Load
  const handleSamplePhotoLoad = async () => {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      const res = await fetch('/api/v1/sample/photo');
      if (!res.ok) throw new Error('Could not fetch sample photo');
      const blob = await res.blob();
      const file = new File([blob], 'silk_board_sample.jpg', { type: 'image/jpeg' });
      await handlePhotoUpload(file);
    } catch (e: any) {
      setErrorMessage(e.message || 'Failed to load sample photo');
      setIsLoading(false);
    }
  };

  // Handle Sample Video Load
  const handleSampleVideoLoad = async () => {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      const res = await fetch('/api/v1/sample/video');
      if (!res.ok) throw new Error('Could not fetch sample video');
      const blob = await res.blob();
      const file = new File([blob], 'sample_traffic.mp4', { type: 'video/mp4' });
      await handleVideoUpload(file);
    } catch (e: any) {
      setErrorMessage(e.message || 'Failed to load sample video');
      setIsLoading(false);
    }
  };

  // Toggle Emergency Green Corridor
  const toggleGreenCorridor = async () => {
    const nextState = !isGreenCorridor;
    setIsGreenCorridor(nextState);
    try {
      await fetch(`/api/v1/incidents/emergency-override?enable=${nextState}`, {
        method: 'POST',
      });
    } catch (e) {}
  };

  return (
    <div className="app-shell">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isGreenCorridor={isGreenCorridor}
        onToggleGreenCorridor={toggleGreenCorridor}
        serverStatus={serverStatus}
        onCheckHealth={checkServer}
      />

      <main className="main-content">
        {errorMessage && (
          <div className="error-banner glass-panel">
            <span>⚠️ {errorMessage}</span>
            <button className="close-btn" onClick={() => setErrorMessage(null)}>
              ✕
            </button>
          </div>
        )}

        {/* PHOTO DETECTION TAB */}
        {activeTab === 'photo' && (
          <div className="tab-pane">
            <UploadDropzone
              mode="photo"
              confidenceThreshold={confidenceThreshold}
              setConfidenceThreshold={setConfidenceThreshold}
              onFileUpload={handlePhotoUpload}
              onSampleLoad={handleSamplePhotoLoad}
              isLoading={isLoading}
            />

            {photoResult && (
              <PhotoResultsView
                result={photoResult}
                originalImageSrc={originalImageSrc}
              />
            )}
          </div>
        )}

        {/* VIDEO TRACKING TAB */}
        {activeTab === 'video' && (
          <div className="tab-pane">
            <UploadDropzone
              mode="video"
              confidenceThreshold={confidenceThreshold}
              setConfidenceThreshold={setConfidenceThreshold}
              onFileUpload={handleVideoUpload}
              onSampleLoad={handleSampleVideoLoad}
              isLoading={isLoading}
            />

            {videoResult && (
              <VideoResultsView
                result={videoResult}
                videoSrc={videoSrc}
                onRescan={() => {
                  if (lastVideoFile) handleVideoUpload(lastVideoFile);
                  else handleSampleVideoLoad();
                }}
              />
            )}
          </div>
        )}

        {/* LIVE CCTV STREAM TAB */}
        {activeTab === 'live' && (
          <div className="tab-pane">
            <LiveStreamView
              isGreenCorridor={isGreenCorridor}
              onToggleGreenCorridor={toggleGreenCorridor}
            />
          </div>
        )}
      </main>

      <footer className="footer-bar glass-panel font-mono">
        <div>
          <span>BGTMS AI v2.0 • SSDLite320-MobileNetV3 + ByteTrack MOT + Webster Adaptive Engine</span>
        </div>
        <div className="footer-links">
          <span>Target Intersection: Central Silk Board, Bengaluru</span>
          <span>IRC PCU Standard Equivalency Compliant</span>
        </div>
      </footer>
    </div>
  );
};

export default App;
