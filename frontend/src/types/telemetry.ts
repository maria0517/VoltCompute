export type AlertLevel = "red" | "amber" | "green";

export type NodeStatus =
  | "ACCEPTED"
  | "OFFER_PENDING"
  | "REJECTED_THERMAL_SAFETY"
  | "REFUSED"
  | "UNCERTAIN"
  | "ESCALATE_TO_HUMAN"
  | "IN_QUEUE"
  | "STANDBY";

export interface TimelineInfo {
  current_step: number;
  ring_cycle: number;
  virtual_index: number;
  hours_in_month: number;
  month_label: string;
  current_timestamp: string;
  current_hour_str: string;
  slider_start_label: string;
  slider_end_label: string;
  train_rows: number;
}

export interface DashboardKPIs {
  raw_injection_mw: number;
  actual_injection_mw: number;
  predicted_injection_mw: number;
  surplus_mw: number;
  actual_surplus_mw: number;
  grid_limit_mw: number;
  confidence: number;
  day_ahead_price_eur: number;
  solar_irradiance_wm2: number;
  ambient_temp_c: number;
  delivered_mw: number;
  revenue_eur: number;
  cumulative_profit_eur: number;
  cumulative_mwh: number;
  live_window_mae: number;
  is_dispatched: boolean;
  ai_tokens_millions: number;
  co2_saved_tons: number;
}

export interface BannerInfo {
  level: AlertLevel;
  text: string;
}

export interface ExplainabilityInfo {
  top_shap_features: Record<string, number>;
  anomalies: string[];
  copilot_reasoning: string;
  min_surplus_threshold_mw: number;
  max_pzu_price_eur: number;
  confidence_threshold: number;
}

export interface ActiveOffer {
  node_id: string;
  node_name: string;
  node_short: string;
  node_color: string;
  per_step: Record<string, number>;
  steps: number[];
  total_mwh: number;
  match_score: number;
  price_eur_mwh: number;
  reference_price_eur_mwh: number;
  buyer_savings_eur: number;
  ppc_revenue_eur: number;
}

export interface OfferState {
  mode: string;
  lead_hours: number;
  offer_steps: number[];
  table_steps: number[];
  offer: ActiveOffer | null;
  offers_exhausted: boolean;
  pool_status: string;
  fixed_price_eur: number;
  reference_tariff_eur: number;
}

export interface DataCenterCard {
  id: string;
  name: string;
  short: string;
  city: string;
  color: string;
  lat: number;
  lon: number;
  waypoints: [number, number][];
  workload: string;
  rank: number;
  match_score: number;
  volume_fit: number;
  acceptance_rate: number;
  thermal_penalty: number;
  eligible: boolean;
  temp_c: number;
  thermal_limit_c: number;
  free_capacity_mw: number;
  base_load_mw: number;
  max_capacity_mw: number;
  allocated_mwh: number;
  offers_sent: number;
  offers_accepted: number;
  status: NodeStatus;
}

export interface ChartPoint {
  v_idx: number;
  timestamp: string;
  short_time: string;
  is_current: boolean;
  is_future: boolean;
  real_injection_mw: number | null;
  predicted_injection_mw: number;
  after_dispatch_mw: number | null;
  predicted_surplus_mw: number;
  delivered_mw: number;
  grid_limit_mw: number;
  confidence: number;
}

export interface SolarParkNode {
  id: string;
  name: string;
  short: string;
  lat: number;
  lon: number;
}

export interface MapTopology {
  solar_park: SolarParkNode;
  backbone_400kv: [number, number][];
}

export interface DashboardStateResponse {
  timeline: TimelineInfo;
  kpis: DashboardKPIs;
  banner: BannerInfo;
  explainability: ExplainabilityInfo;
  offer_state: OfferState;
  datacenters: DataCenterCard[];
  chart_points: ChartPoint[];
  y_axis_max: number;
  map_topology: MapTopology;
  toast?: string;
}
