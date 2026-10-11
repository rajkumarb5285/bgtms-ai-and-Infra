import React, { useState, useRef } from 'react';
import { Image as ImageIcon, Film, Sliders, Sparkles, Loader2 } from 'lucide-react';

interface UploadDropzoneProps {
  mode: 'photo' | 'video';
  confidenceThreshold: number;
  setConfidenceThreshold: (val: number) => void;
  onFileUpload: (file: File) => void;
  onSampleLoad: () => void;
  isLoading: boolean;
}

export const UploadDropzone: React.FC<UploadDropzoneProps> = ({
  mode,
  confidenceThreshold,
  setConfidenceThreshold,
  onFileUpload,
  onSampleLoad,
  isLoading,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileUpload(e.target.files[0]);
    }
  };

  return (
    <div className="upload-controls-container">
      {/* Upload Drop Zone Card */}
      <div
        className={`dropzone-card glass-panel ${isDragging ? 'dropzone-active' : ''} ${isLoading ? 'dropzone-loading' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isLoading && fileInputRef.current?.click()}
      >
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          accept={mode === 'photo' ? 'image/*' : 'video/*'}
          onChange={handleFileChange}
        />

        {isLoading ? (
          <div className="loading-state-wrapper">
            <Loader2 className="spinner-icon" size={44} />
            <h3 className="loading-title">Neural Engine Running Inference...</h3>
            <p className="loading-subtitle">
              {mode === 'photo'
                ? 'Processing SSDLite320-MobileNetV3 backbone, IRC PCU equivalencies, and incident checks'
                : 'Decoding video frames, ByteTrack Kalman association, and traffic timeline synthesis'}
            </p>
          </div>
        ) : (
          <div className="dropzone-content">
            <div className="upload-icon-wrapper">
              {mode === 'photo' ? <ImageIcon size={32} /> : <Film size={32} />}
            </div>
            <h3 className="dropzone-title">
              Drop your {mode === 'photo' ? 'traffic photo' : 'traffic CCTV video'} here, or{' '}
              <span className="browse-highlight">browse files</span>
            </h3>
            <p className="dropzone-desc">
              {mode === 'photo'
                ? 'Supports JPG, PNG, WEBP (over-head road angles, intersection footage)'
                : 'Supports MP4, AVI, MOV (multi-object tracking & timeline velocity analysis)'}
            </p>

            <div className="dropzone-action-bar" onClick={(e) => e.stopPropagation()}>
              <button
                type="button"
                className="sample-btn"
                onClick={onSampleLoad}
                disabled={isLoading}
              >
                <Sparkles size={16} />
                <span>Load Sample Indian Traffic {mode === 'photo' ? 'Photo' : 'Video'}</span>
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Model Parameters & Tuning Card */}
      <div className="settings-card glass-panel">
        <div className="card-header">
          <div className="header-title-row">
            <Sliders size={18} className="icon-cyan" />
            <h4>AI Model Parameters</h4>
          </div>
          <span className="badge-model font-mono">SSDLite320 • ONNX / PyTorch</span>
        </div>

        <div className="setting-control-row">
          <div className="slider-label-row">
            <span className="slider-title">Confidence Threshold</span>
            <span className="slider-val-badge font-mono">
              {(confidenceThreshold * 100).toFixed(0)}%
            </span>
          </div>
          <input
            type="range"
            min="0.10"
            max="0.85"
            step="0.05"
            value={confidenceThreshold}
            onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
            className="custom-range-slider"
          />
          <div className="slider-ticks">
            <span>10% (High Recall)</span>
            <span>Default (35%)</span>
            <span>85% (High Precision)</span>
          </div>
        </div>

        <div className="taxonomy-pills-row">
          <span className="tax-label">Active Vehicle Classes:</span>
          <div className="tax-tags">
            <span className="tag-pill tag-car">Car (1.0 PCU)</span>
            <span className="tag-pill tag-bike">2-Wheeler (0.5 PCU)</span>
            <span className="tag-pill tag-auto">Auto (1.0 PCU)</span>
            <span className="tag-pill tag-bus">Bus (3.0 PCU)</span>
            <span className="tag-pill tag-truck">Truck (3.0 PCU)</span>
            <span className="tag-pill tag-emg">Ambulance (1.5 PCU)</span>
          </div>
        </div>
      </div>
    </div>
  );
};
