import { ShieldAlert, Camera, Video, Radio } from 'lucide-react';

interface NavbarProps {
  activeTab: 'photo' | 'video' | 'live';
  setActiveTab: (tab: 'photo' | 'video' | 'live') => void;
  isGreenCorridor: boolean;
  onToggleGreenCorridor: () => void;
  serverStatus: 'connected' | 'offline' | 'checking';
  onCheckHealth?: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  isGreenCorridor,
  onToggleGreenCorridor,
  serverStatus,
  onCheckHealth,
}) => {
  return (
    <header className="navbar-container glass-panel">
      <div className="brand-group">
        <div className="brand-logo">
          <div className="logo-glow"></div>
          <span className="logo-emoji">🚦</span>
        </div>
        <div className="brand-text">
          <div className="brand-title-row">
            <h1 className="brand-title">BGTMS <span className="title-ai">AI</span></h1>
            <span className="badge-pill badge-primary">M2 PRODUCTION</span>
          </div>
          <p className="brand-subtitle">Bengaluru Intelligent Traffic Management & Incident Intelligence</p>
        </div>
      </div>

      <nav className="tab-navigation">
        <button
          className={`tab-btn ${activeTab === 'photo' ? 'active' : ''}`}
          onClick={() => setActiveTab('photo')}
        >
          <Camera size={16} />
          <span>Photo Detection</span>
          {activeTab === 'photo' && <div className="tab-indicator" />}
        </button>

        <button
          className={`tab-btn ${activeTab === 'video' ? 'active' : ''}`}
          onClick={() => setActiveTab('video')}
        >
          <Video size={16} />
          <span>Video Tracking</span>
          {activeTab === 'video' && <div className="tab-indicator" />}
        </button>

        <button
          className={`tab-btn ${activeTab === 'live' ? 'active' : ''}`}
          onClick={() => setActiveTab('live')}
        >
          <Radio size={16} />
          <span>Live CCTV Stream</span>
          <span className="live-dot" />
          {activeTab === 'live' && <div className="tab-indicator" />}
        </button>
      </nav>

      <div className="action-status-group">
        <div
          className="status-indicator-box"
          onClick={onCheckHealth}
          title="Click to re-check backend connection status"
          style={{ cursor: 'pointer' }}
        >
          <span className={`status-dot ${serverStatus === 'connected' ? 'dot-online' : 'dot-offline'}`} />
          <span className="status-text font-mono">
            {serverStatus === 'connected' ? 'FASTAPI :8000' : 'BACKEND OFFLINE (RETRY)'}
          </span>
        </div>

        <button
          className={`corridor-btn ${isGreenCorridor ? 'corridor-active' : ''}`}
          onClick={onToggleGreenCorridor}
        >
          <ShieldAlert size={16} />
          <span>{isGreenCorridor ? 'CLEAR CORRIDOR' : 'EMERGENCY PREEMPTION'}</span>
        </button>
      </div>
    </header>
  );
};
