/** Backend API client — all calls go to /api (proxied to FastAPI in dev). */

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: init?.body ? { "Content-Type": "application/json" } : undefined,
    ...init,
  });
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      const data = await res.json();
      detail = data?.detail ?? detail;
    } catch {
      /* non-JSON error */
    }
    throw new ApiError(detail, res.status);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "POST", body: body ? JSON.stringify(body) : undefined }),
  patch: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "PATCH", body: JSON.stringify(body) }),
  delete: <T>(path: string) => request<T>(path, { method: "DELETE" }),
};

export type ModelInfo = { id: string; revision: string; architecture?: string | null; license?: string | null; total_disk_bytes?: number | null };

export type DatasetMeta = {
  id: string;
  name: string;
  source: string;
  format: string;
  schema_type: string | null;
  example_count: number;
  status: string;
  created_at: string;
  project_id: string | null;
};

export type Project = {
  id: string;
  name: string;
  description: string;
  spec: Record<string, unknown> | null;
  spec_approved: boolean;
  base_model: Record<string, unknown> | null;
  dataset_id: string | null;
  status: string;
  activity: { kind: string; message: string; ts: string }[] | null;
  created_at?: string;
};

export type JobMetric = {
  step: number;
  epoch: number | null;
  loss: number | null;
  eval_loss: number | null;
  learning_rate: number | null;
  ts: string;
};

export type Job = {
  id: string;
  project_id: string | null;
  dataset_id: string | null;
  mode: string;
  status: string;
  config: Record<string, any> | null;
  error: string | null;
  exit_code: number | null;
  progress_step: number;
  total_steps: number;
  run_dir: string | null;
  created_at?: string;
};

export const fmtBytes = (n: number | null | undefined): string | null => {
  if (n == null) return null;
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
};
