export type Health = {
  status: string;
  version: string;
  uptime_s: number;
  device: string;
  device_name: string;
  segmentation_ready: boolean;
  classification_ready: boolean;
};

export type ModelInfo = {
  id: string;
  task: "segmentation" | "classification";
  arch: string;
  dataset: string;
  image_size: number;
  threshold: number;
  class_names: string[];
  metrics: Record<string, number>;
  run_id: string;
  created_at: string;
  git_commit: string;
  parameters: number;
  loaded: boolean;
};

export type RegistryVersion = {
  version: string;
  stage: string;
  status: string;
  run_id: string;
  created: number;
  metrics: Record<string, number>;
};

export type RegistryModel = {
  name: string;
  champion: string | null;
  versions: RegistryVersion[];
};

export type ModelsResponse = {
  device: string;
  device_name: string;
  model_dir: string;
  models: ModelInfo[];
  segmentation_ready: boolean;
  classification_ready: boolean;
  load_errors: { file: string; error: string }[];
  registry: RegistryModel[];
};

export type SegResponse = {
  prediction_id: string;
  task: "segmentation";
  model_id: string;
  model_version: string;
  threshold: number;
  tta: boolean;
  latency_ms: number;
  lesion_fraction: number;
  mean_probability: number;
  uncertainty_mean: number;
  width: number;
  height: number;
  preview_base64: string;
  overlay_base64: string;
  mask_base64: string;
  uncertainty_base64: string;
  features: Record<string, number>;
};

export type ClfResponse = {
  prediction_id: string;
  task: "classification";
  model_id: string;
  model_version: string;
  latency_ms: number;
  predicted_label: string;
  predicted_index: number;
  confidence: number;
  margin: number;
  labels: string[];
  probabilities: number[];
  width: number;
  height: number;
  preview_base64: string;
  gradcam_base64: string | null;
  features: Record<string, number>;
};

export type RunSummary = {
  run_id: string;
  run_name: string;
  experiment: string;
  status: string;
  start_time: number;
  duration_s: number | null;
  metrics: Record<string, number>;
  params: Record<string, string>;
  tags: Record<string, string>;
};

export type RunDetail = {
  run_id: string;
  run_name: string;
  status: string;
  experiment_id: string;
  params: Record<string, string>;
  metrics: Record<string, number>;
  tags: Record<string, string>;
  history: Record<string, { step: number; value: number; timestamp: number }[]>;
};

export type DriftFeature = {
  psi: number;
  status: "stable" | "watch" | "alert";
  reference_mean: number;
  current_mean: number;
  n: number;
};

export type MonitoringSummary = {
  total_predictions: number;
  by_task: Record<string, { predictions: number; avg_latency_ms: number; with_feedback: number }>;
  recent_latency_p95_ms: number;
  feedback: Record<string, number>;
  labels: { segmentation: Record<string, number>; classification: Record<string, number> };
  drift: Record<
    string,
    {
      status: string;
      features: Record<string, DriftFeature>;
      window: number;
      min_samples?: number;
      has_reference?: boolean;
    }
  >;
  recent: {
    id: string;
    ts: number;
    task: string;
    model_id: string;
    model_version: string;
    filename: string | null;
    latency_ms: number;
    label: string | null;
    confidence: number | null;
    feedback: { verdict: string; comment: string } | null;
  }[];
};

export type DemoCase = {
  task: string;
  name: string;
  image_base64: string;
  width: number;
  height: number;
};

export type Job = {
  id: string;
  config: string;
  overrides: string[];
  status: "running" | "succeeded" | "failed" | "cancelled";
  started_at: number;
  finished_at: number | null;
  returncode: number | null;
  log_lines: number;
  log?: string[];
  log_offset?: number;
};

export type SystemInfo = {
  version: string;
  python: string;
  platform: string;
  torch: string;
  cuda: string | null;
  fastapi: string;
  device: string;
  device_name: string;
  models_loaded: number;
  tracking: { tracking_uri?: string; backend?: string; ui_url?: string | null; experiments?: string[] };
  datasets: Record<string, { dataset?: string; splits?: Record<string, { count: number }>; class_names?: string[]; status?: string }>;
  uptime_s: number;
};

async function request<T>(input: string, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(message);
  }
  return (await response.json()) as T;
}

const jsonInit = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<Health>("/api/health"),
  system: () => request<SystemInfo>("/api/system"),
  models: () => request<ModelsResponse>("/api/models"),
  refreshModels: () => request<ModelsResponse>("/api/models/refresh", { method: "POST" }),
  promote: (registry_name: string, run_id: string) =>
    request<{ name: string; version: string }>("/api/models/promote", jsonInit({ registry_name, run_id })),
  experiments: (limit = 30) => request<{ runs: RunSummary[] }>(`/api/experiments?limit=${limit}`),
  run: (runId: string) => request<RunDetail>(`/api/experiments/${runId}`),
  monitoring: () => request<MonitoringSummary>("/api/monitoring/summary"),
  feedback: (prediction_id: string, verdict: string) =>
    request<{ status: string }>("/api/feedback", jsonInit({ prediction_id, verdict })),
  demo: (task: string) => request<DemoCase>(`/api/demo/${task}`),
  predictSegmentation: (file: File | Blob, modelId: string | null, tta: boolean) => {
    const form = new FormData();
    form.append("file", file, file instanceof File ? file.name : "demo.png");
    if (modelId) form.append("model_id", modelId);
    form.append("tta", String(tta));
    return request<SegResponse>("/api/predict/segmentation", { method: "POST", body: form });
  },
  predictClassification: (file: File | Blob, modelId: string | null) => {
    const form = new FormData();
    form.append("file", file, file instanceof File ? file.name : "demo.png");
    if (modelId) form.append("model_id", modelId);
    return request<ClfResponse>("/api/predict/classification", { method: "POST", body: form });
  },
  jobs: () => request<{ jobs: Job[] }>("/api/pipeline/jobs"),
  startTraining: (config: string, overrides: string[]) =>
    request<Job>("/api/pipeline/train", jsonInit({ config, overrides })),
  job: (id: string, offset = 0) => request<Job>(`/api/pipeline/jobs/${id}?offset=${offset}`),
  cancelJob: (id: string) => request<Job>(`/api/pipeline/jobs/${id}/cancel`, { method: "POST" }),
};

export const imgSrc = (b64: string) => `data:image/png;base64,${b64}`;

export function formatMs(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value < 1000 ? `${value.toFixed(1)} ms` : `${(value / 1000).toFixed(2)} s`;
}

export function formatPct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function timeAgo(ts: number | null | undefined): string {
  if (!ts) return "—";
  const seconds = Math.max(1, Math.round(Date.now() / 1000 - ts));
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`;
  return `${Math.round(seconds / 86400)}d ago`;
}
