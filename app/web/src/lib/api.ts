/**
 * Thin fetch wrapper for the AceJAM REST API. All endpoints live on the same
 * origin as the React app (mounted at /v2 from FastAPI). In dev, Vite proxies
 * /api and /media to the Python server at 127.0.0.1:7860.
 */

export class ApiError extends Error {
  status: number;
  payload: unknown;
  constructor(message: string, status: number, payload?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

async function http<T = unknown>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !headers.has("Content-Type") && !(init.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  headers.set("Accept", "application/json");
  const res = await fetch(path, { ...init, headers });
  const text = await res.text();
  let payload: unknown = text;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      // keep as text
    }
  }
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    if (payload && typeof payload === "object" && "error" in (payload as Record<string, unknown>)) {
      const e = (payload as Record<string, unknown>).error;
      if (typeof e === "string" && e) msg = e;
    }
    throw new ApiError(msg, res.status, payload);
  }
  return payload as T;
}

export const api = {
  get: <T = unknown>(path: string) => http<T>(path),
  post: <T = unknown>(path: string, body?: unknown) =>
    http<T>(path, {
      method: "POST",
      body: body instanceof FormData ? body : JSON.stringify(body ?? {}),
    }),
};

// ---- Endpoint shortcuts ----

export type WizardMode =
  | "simple"
  | "custom"
  | "song"
  | "album"
  | "cover"
  | "repaint"
  | "extract"
  | "lego"
  | "complete"
  | "news";

export interface PromptAssistantRunRequest {
  mode: WizardMode;
  user_prompt: string;
  current_payload?: Record<string, unknown>;
  planner_model?: string;
  planner_lm_provider?: string;
  planner_ollama_model?: string;
}

export interface PromptAssistantRunResponse {
  success: boolean;
  payload?: Record<string, unknown>;
  warnings?: string[];
  paste_blocks?: Array<{ label?: string; content: string }>;
  error?: string;
  raw_response?: string;
}

export const promptAssistantRun = (body: PromptAssistantRunRequest) =>
  api.post<PromptAssistantRunResponse>("/api/prompt-assistant/run", body);

export const getConfig = () => api.get<Record<string, unknown>>("/api/config");

// ---- Local LLM catalog ----------------------------------------------------
//
// `/api/local-llm/catalog` returns the full mixed catalog: settings, providers,
// per-provider sub-catalogs, and a flattened `details[]` with rich metadata
// (kind, capabilities, image_generation, profile, etc). All model dropdowns
// in the React UI consume this single endpoint and filter client-side.

export type LLMProvider = "ollama" | "lmstudio" | "ace_step_lm";

export interface LLMModelProfile {
  label?: string;
  dropdown_label?: string;
  summary?: string;
  quality?: string;
  speed?: string;
  vram?: string;
  notes?: string;
  warnings?: string[];
  source_urls?: string[];
}

export interface LLMModelDetail {
  key: string;
  name: string;
  display_name?: string;
  provider: LLMProvider;
  kind?: "chat" | "embedding" | string;
  type?: string;
  capabilities?: string[] | Record<string, unknown>;
  image_generation?: boolean;
  vision?: boolean;
  size_gb?: number;
  parameter_size?: string;
  format?: string;
  quantization_level?: string;
  loaded?: boolean;
  status?: string;
  profile?: LLMModelProfile;
  mlx_preferred?: boolean;
}

export interface LLMProviderEntry {
  id: LLMProvider;
  label: string;
  host: string;
  ready: boolean;
}

export interface LLMSubCatalog {
  success: boolean;
  ready: boolean;
  host?: string;
  models: string[];
  chat_models: string[];
  embedding_models: string[];
  image_models: string[];
  details: LLMModelDetail[];
  error?: string;
}

export interface LLMCatalogResponse {
  success: boolean;
  settings: {
    provider?: LLMProvider;
    chat_model?: string;
    embedding_provider?: LLMProvider;
    embedding_model?: string;
    art_provider?: LLMProvider;
    art_model?: string;
    [k: string]: unknown;
  };
  providers: LLMProviderEntry[];
  catalogs: Partial<Record<LLMProvider, LLMSubCatalog>>;
  details: LLMModelDetail[];
  models: string[];
  chat_models: string[];
  embedding_models: string[];
  image_models: string[];
}

export const getLLMCatalog = () =>
  api.get<LLMCatalogResponse>("/api/local-llm/catalog");

function capList(d: LLMModelDetail): string[] {
  return Array.isArray(d.capabilities) ? d.capabilities : [];
}

export function chatModelDetails(c: LLMCatalogResponse | undefined): LLMModelDetail[] {
  if (!c) return [];
  return c.details.filter(
    (d) =>
      d.kind === "chat" ||
      capList(d).includes("chat") ||
      // ACE-Step LMs expose capabilities like 'composition'/'metadata' etc; treat them as chat planners
      d.provider === "ace_step_lm",
  );
}

export function imageModelDetails(c: LLMCatalogResponse | undefined): LLMModelDetail[] {
  if (!c) return [];
  return c.details.filter(
    (d) => d.image_generation === true || capList(d).includes("image_generation"),
  );
}

export function embeddingModelDetails(c: LLMCatalogResponse | undefined): LLMModelDetail[] {
  if (!c) return [];
  return c.details.filter((d) => d.kind === "embedding");
}

export const PROVIDER_LABEL: Record<LLMProvider, string> = {
  ollama: "Ollama",
  lmstudio: "LM Studio",
  ace_step_lm: "ACE-Step LM",
};

// ---- Generation ----

export interface GenerateAdvancedResponse {
  success: boolean;
  result_id?: string;
  audio_url?: string;
  audio?: string; // base64 fallback
  title?: string;
  artist_name?: string;
  tags?: string;
  caption?: string;
  lyrics?: string;
  duration?: number;
  bpm?: number;
  key_scale?: string;
  vocal_language?: string;
  art?: { art_id: string; url: string };
  payload_warnings?: string[];
  error?: string;
  [k: string]: unknown;
}

export const generateAdvanced = (body: Record<string, unknown>) =>
  api.post<GenerateAdvancedResponse>("/api/generate_advanced", body);

export const createSample = (body: Record<string, unknown>) =>
  api.post<GenerateAdvancedResponse>("/api/create_sample", body);

// ---- Albums ----

export interface AlbumPlanJobRequest extends Record<string, unknown> {
  concept: string;
  num_tracks?: number;
  track_duration?: number;
  language?: string;
}

export interface AlbumPlanJobResponse {
  success: boolean;
  job_id?: string;
  status?: string;
  error?: string;
}

export const startAlbumPlanJob = (body: AlbumPlanJobRequest) =>
  api.post<AlbumPlanJobResponse>("/api/album/plan/jobs", body);

export const getAlbumPlanJob = (jobId: string) =>
  api.get<{
    success: boolean;
    job?: {
      status: string;
      progress?: number;
      result?: Record<string, unknown>;
      error?: string;
    };
  }>(`/api/album/plan/jobs/${encodeURIComponent(jobId)}`);

// ---- Artwork ----

export interface ArtGenerateRequest extends Record<string, unknown> {
  scope: "single" | "album" | "test";
  title?: string;
  caption?: string;
  prompt?: string;
  album_title?: string;
  album_concept?: string;
  negative_prompt?: string;
  width?: number;
  height?: number;
  steps?: number;
  seed?: number | string;
  model?: string;
  attach_to_result_id?: string;
  attach_to_album_family_id?: string;
}

export interface ArtMetadata {
  art_id: string;
  filename: string;
  url: string;
  scope: string;
  prompt: string;
  width: number;
  height: number;
  model?: string;
  created_at?: string;
}

export const generateArt = (body: ArtGenerateRequest) =>
  api.post<{ success: boolean; art?: ArtMetadata; error?: string }>(
    "/api/art/generate",
    body,
  );

// ---- Library / community ----

export interface SongMeta {
  song_id?: string;
  result_id?: string;
  title?: string;
  artist_name?: string;
  tags?: string | string[];
  caption?: string;
  lyrics?: string;
  duration?: number;
  bpm?: number;
  key_scale?: string;
  vocal_language?: string;
  audio_url?: string;
  art?: ArtMetadata;
  song_model?: string;
  created_at?: string;
}

export const listCommunity = () =>
  api.get<{ success: boolean; songs: SongMeta[] }>("/api/community");

// ---- Uploads ----

export const uploadFile = (file: File) => {
  const fd = new FormData();
  fd.append("file", file);
  return api.post<{ success: boolean; upload_id?: string; url?: string; error?: string }>(
    "/api/uploads",
    fd,
  );
};

// ---- Misc ----

export const getStatus = () =>
  api.get<{ success: boolean; status?: string; runtime?: Record<string, unknown> }>(
    "/api/status",
  );

export const deleteSong = (song_id: string) =>
  api.post<{ success: boolean; error?: string }>("/api/delete_song", { song_id });
