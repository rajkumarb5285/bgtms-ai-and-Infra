export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  class_id: number;
  class_name: string;
}

export interface DetectionItem {
  class_name: string;
  confidence: number;
  bbox?: number[];
  x1?: number;
  y1?: number;
  x2?: number;
  y2?: number;
  pcu?: number;
  speed_kmh?: number;
  lane_id?: number;
}

export interface LaneMetric {
  lane_id: number;
  approach_id: string;
  vehicle_count: number;
  pcu_count: number;
  density_veh_km: number;
  occupancy_pct: number;
  flow_pcu_hr: number;
  avg_speed_kmh: number;
  queue_length_m: number;
  waiting_vehicles: number;
}

export interface ApproachState {
  approach_id: string;
  approach_name: string;
  total_vehicles: number;
  total_pcu: number;
  average_speed_kmh: number;
  queue_length_m: number;
  congestion_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  lanes: LaneMetric[];
}

export interface IncidentAlert {
  incident_id: string;
  incident_type: string;
  severity: 'INFO' | 'WARNING' | 'CRITICAL';
  timestamp: number;
  track_id?: number;
  approach_id?: string;
  description: string;
  details?: Record<string, any>;
  video_time_sec?: number;
}

export interface SignalPhase {
  phase_id: number;
  name: string;
  green_duration_sec: number;
  yellow_duration_sec: number;
  all_red_duration_sec: number;
  allocated_approaches: string[];
}

export interface SignalPlan {
  junction_id: string;
  cycle_time_sec: number;
  current_phase_id: number;
  time_in_phase_sec: number;
  phases: SignalPhase[];
  is_green_corridor: boolean;
  green_corridor_approach?: string;
}

export interface PhotoAnalysisResult {
  status: string;
  filename: string;
  width: number;
  height: number;
  detections_count: number;
  total_pcu: number;
  congestion_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  average_speed_kmh: number;
  class_breakdown: Record<string, number>;
  detections: DetectionItem[];
  approach_state?: ApproachState;
  incidents?: IncidentAlert[];
  signal_plan?: SignalPlan;
  annotated_image: string;
  latency_ms: number;
  continuous_learning?: {
    capture_id?: string;
    timestamp?: string;
    s3_uri?: string;
    is_hard_example?: boolean;
    detections_count?: number;
  };
}

export interface VideoTimelinePoint {
  second: number;
  vehicles: number;
  pcu: number;
}

export interface VideoAnalysisResult {
  status: string;
  filename: string;
  duration_sec: number;
  processed_frames: number;
  total_unique_vehicles: number;
  total_unique_pcu: number;
  peak_pcu: number;
  class_breakdown: Record<string, number>;
  timeline: VideoTimelinePoint[];
  incidents: IncidentAlert[];
  peak_snapshot?: string;
  latency_ms: number;
  continuous_learning?: {
    capture_id?: string;
    timestamp?: string;
    s3_uri?: string;
    is_hard_example?: boolean;
    detections_count?: number;
  };
}

export interface LiveTelemetryPayload {
  frame_index: number;
  latency_ms: number;
  fps: number;
  detections_count: number;
  approach_state?: ApproachState;
  signal_plan?: SignalPlan;
  new_incidents?: IncidentAlert[];
  total_incidents_count?: number;
}
