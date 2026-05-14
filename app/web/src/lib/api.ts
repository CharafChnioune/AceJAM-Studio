/**
 * Thin fetch wrapper for the MLX Media REST API. All endpoints live on the same
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
  | "news"
  | "image"
  | "video";

export interface PromptAssistantRunRequest {
  mode: WizardMode;
  user_prompt: string;
  current_payload?: Record<string, unknown>;
  planner_model?: string;
  planner_lm_provider?: string;
  planner_ollama_model?: string;
  embedding_provider?: string;
  embedding_lm_provider?: string;
  embedding_model?: string;
}

export interface PromptAssistantRunResponse {
  success: boolean;
  payload?: Record<string, unknown>;
  warnings?: string[] | string | null;
  paste_blocks?: Array<{ label?: string; content?: string; text?: string }> | string | null;
  error?: string;
  raw_response?: string;
}

export const promptAssistantRun = (body: PromptAssistantRunRequest) =>
  api.post<PromptAssistantRunResponse>("/api/prompt-assistant/run", body);

export const getConfig = () => api.get<Record<string, unknown>>("/api/config");

export interface AudioStyleProfile {
  key: string;
  style_profile?: string;
  label?: string;
  caption_tags?: string;
  lyrics_section_tags?: Record<string, string>;
  lyrics?: string;
  test_lyrics?: string;
  bpm?: number | null;
  keyscale?: string;
  timesignature?: string;
  vocal_language?: string;
  default?: boolean;
}

export const getAudioStyleProfiles = () =>
  api.get<{ success: boolean; profiles?: AudioStyleProfile[] }>("/api/audio/style-profiles");

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

export interface GenerationJob {
  id: string;
  task_id?: string;
  kind?: "generation";
  state?: string;
  status?: string;
  stage?: string;
  progress?: number;
  created_at?: string;
  updated_at?: string;
  finished_at?: string;
  payload?: Record<string, unknown>;
  payload_summary?: Record<string, unknown>;
  result?: GenerateAdvancedResponse;
  result_summary?: Record<string, unknown>;
  warnings?: string[];
  logs?: string[];
  error?: string;
}

export interface GenerationJobResponse {
  success: boolean;
  job_id?: string;
  task_id?: string;
  job?: GenerationJob;
  error?: string;
}

export const startGenerationJob = (body: Record<string, unknown>) =>
  api.post<GenerationJobResponse>("/api/generation/jobs", body);

export const getGenerationJob = (jobId: string) =>
  api.get<GenerationJobResponse>(`/api/generation/jobs/${encodeURIComponent(jobId)}`);

// ---- Song batches ----

export interface SongBatchSong {
  index?: number;
  track_number?: number;
  title?: string;
  state?: string;
  status?: string;
  progress?: number;
  generation_job_id?: string;
  payload?: Record<string, unknown>;
  payload_summary?: Record<string, unknown>;
  result?: GenerateAdvancedResponse | Record<string, unknown> | null;
  result_summary?: Record<string, unknown>;
  audio_urls?: string[];
  error?: string;
  started_at?: string;
  finished_at?: string;
}

export interface SongBatchJob {
  id: string;
  kind?: "song_batch";
  state?: string;
  status?: string;
  stage?: string;
  progress?: number;
  batch_title?: string;
  payload?: Record<string, unknown>;
  payload_summary?: Record<string, unknown>;
  songs?: SongBatchSong[];
  logs?: string[];
  errors?: string[];
  current_song?: number;
  total_songs?: number;
  completed_songs?: number;
  failed_songs?: number;
  remaining_songs?: number;
  child_generation_job_id?: string;
  created_at?: string;
  started_at?: string;
  finished_at?: string;
  updated_at?: string;
  error?: string;
}

export interface SongBatchJobResponse {
  success: boolean;
  job_id?: string;
  job?: SongBatchJob;
  jobs?: SongBatchJob[];
  error?: string;
}

export const startSongBatchJob = (body: Record<string, unknown>) =>
  api.post<SongBatchJobResponse>("/api/song-batches/jobs", body);

export const getSongBatchJob = (jobId: string) =>
  api.get<SongBatchJobResponse>(`/api/song-batches/jobs/${encodeURIComponent(jobId)}`);

export const listSongBatchJobs = () =>
  api.get<SongBatchJobResponse>("/api/song-batches/jobs");

// ---- LoRA adapters -------------------------------------------------------

export interface LoraAdapter {
  name: string;
  display_name?: string;
  label?: string;
  path: string;
  adapter_type?: string;
  trigger_tag?: string;
  trigger_tag_raw?: string;
  generation_trigger_tag?: string;
  trigger_aliases?: string[];
  trigger_source?: string;
  trigger_candidates?: string[];
  language?: string;
  model_variant?: string;
  song_model?: string;
  is_loadable?: boolean;
  generation_loadable?: boolean;
  source?: string;
  updated_at?: string;
  metadata?: Record<string, unknown>;
}

export const getLoraAdapters = () =>
  api.get<{ success: boolean; adapters: LoraAdapter[] }>("/api/lora/adapters");

// ---- Albums ----

export interface AlbumPlanJobRequest extends Record<string, unknown> {
  concept: string;
  num_tracks?: number;
  track_duration?: number;
  duration_mode?: "ai_per_track" | "fixed";
  album_writer_mode?: "per_track_writer_loop";
  language?: string;
  tracks?: Record<string, unknown>[];
}

export interface AlbumPlanJobResponse {
  success: boolean;
  job_id?: string;
  job?: Record<string, unknown>;
  status?: string;
  error?: string;
  message?: string;
  ollama_pull_started?: boolean;
  ollama_model?: string;
  ollama_pull_job?: Record<string, unknown>;
  logs?: string[];
}

export const startAlbumPlanJob = (body: AlbumPlanJobRequest) =>
  api.post<AlbumPlanJobResponse>("/api/album/plan/jobs", body);

export const startAlbumJob = (body: AlbumPlanJobRequest) =>
  api.post<AlbumPlanJobResponse>("/api/album/jobs", body);

export const getAlbumPlanJob = (jobId: string) =>
  api.get<{
    success: boolean;
    job?: {
      status?: string;
      state?: string;
      stage?: string;
      current_task?: string;
      current_agent?: string;
      current_track?: number | string;
      total_tracks?: number;
      completed_tracks?: number;
      remaining_tracks?: number;
      waiting_on_llm?: boolean;
      llm_provider?: string;
      llm_model?: string;
      llm_wait_elapsed_s?: number;
      last_update_at?: string;
      logs?: string[];
      progress?: number;
      result?: Record<string, unknown>;
      error?: string;
      errors?: string[];
    };
  }>(`/api/album/jobs/${encodeURIComponent(jobId)}`);

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

export interface LibraryItem extends SongMeta {
  id: string;
  kind: "audio" | "image" | "video";
  source: "song" | "result" | "mlx-video" | string;
  deletable?: boolean;
  audio_id?: string;
  filename?: string;
  download_url?: string;
  image_url?: string;
  thumbnail_url?: string;
  video_url?: string;
  url?: string;
  poster_url?: string;
  prompt?: string;
  model_label?: string;
  action?: string;
  width?: number;
  height?: number;
  recommended?: boolean;
  use_lora?: boolean;
  lora_scale?: number;
  raw?: Record<string, unknown>;
}

export interface LibraryResponse {
  success: boolean;
  items: LibraryItem[];
  counts: {
    all: number;
    songs: number;
    results: number;
    images: number;
    videos: number;
    audio: number;
  };
}

export const listLibrary = () =>
  api.get<LibraryResponse>("/api/library");

export interface LibraryDeleteTarget {
  id: string;
  kind: "song" | "result-audio" | "image" | "video";
  result_id?: string;
  song_id?: string;
  audio_id?: string;
  filename?: string;
}

export const deleteLibraryItem = (body: LibraryDeleteTarget & {
  confirm: "DELETE";
}) => api.post<{ success: boolean; deleted?: Record<string, unknown>; error?: string }>("/api/library/delete", body);

export const deleteLibraryItems = (body: {
  items: LibraryDeleteTarget[];
  confirm: "DELETE";
}) => api.post<{ success: boolean; deleted?: Record<string, unknown>; errors?: Array<Record<string, unknown>>; error?: string }>("/api/library/delete", body);

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

// ---- LoRA dataset auto-label background job ----

export interface LoraAutolabelLabel {
  path?: string;
  filename: string;
  lyrics?: string;
  caption?: string;
  genre?: string;
  style_profile?: string;
  caption_tags?: string;
  genre_label_source?: string;
  genre_confidence?: number | string;
  genre_reason?: string;
  genre_label_provider?: string;
  genre_label_model?: string;
  genre_label_error?: string;
  language?: string;
  bpm?: number | string;
  keyscale?: string;
  label_source?: string;
  error?: string;
  lyrics_path?: string;
  metadata_path?: string;
}

export interface LoraAutolabelJob {
  id: string;
  state?: string;
  status?: string;
  progress?: number;
  processed?: number;
  total?: number;
  succeeded?: number;
  failed?: number;
  current_file?: string;
  logs?: string[];
  errors?: string[];
  labels?: LoraAutolabelLabel[];
  dataset_id?: string;
}

export const startLoraAutolabelJob = (body: {
  dataset_id: string;
  ace_lm_model?: string;
  language?: string;
  song_model?: string;
  skip_existing?: boolean;
  genre?: string;
  genre_label_mode?: "ai_auto" | "manual_global" | "metadata_musicbrainz";
  genre_label_provider?: LLMProvider;
  genre_label_model?: string;
  overwrite_existing_labels?: boolean;
}) =>
  api.post<{ success: boolean; job_id?: string; job?: LoraAutolabelJob; error?: string }>(
    "/api/lora/dataset/autolabel/jobs",
    body,
  );

export const getLoraAutolabelJob = (jobId: string) =>
  api.get<{ success: boolean; job?: LoraAutolabelJob; error?: string }>(
    `/api/lora/dataset/autolabel/jobs/${encodeURIComponent(jobId)}`,
  );

// ---- MFLUX image studio --------------------------------------------------

export interface MfluxStatus {
  success: boolean;
  ready: boolean;
  platform?: string;
  arch?: string;
  apple_silicon?: boolean;
  mlx_available?: boolean;
  mflux_available?: boolean;
  cli_available?: boolean;
  blocking_reason?: string;
  data_dir?: string;
  results_dir?: string;
  uploads_dir?: string;
  datasets_dir?: string;
  lora_dir?: string;
  commands?: Record<string, string | null>;
  command_help?: Record<string, { available: boolean; help_ok: boolean; reason?: string }>;
  action_readiness?: Record<string, {
    label?: string;
    ready: boolean;
    commands?: string[];
    available_commands?: string[];
    missing_commands?: string[];
    models?: string[];
    reason?: string;
  }>;
}

export interface MfluxModel {
  id: string;
  label: string;
  preset: string;
  family?: string;
  family_label?: string;
  size?: string;
  command?: string;
  edit_command?: string;
  model_arg?: string;
  quantization_default?: number;
  default_steps?: number;
  default_width?: number;
  default_height?: number;
  capabilities?: string[];
  trainable?: boolean;
  description?: string;
}

export interface MfluxModelsResponse {
  success: boolean;
  status?: MfluxStatus;
  models: MfluxModel[];
  presets?: Record<string, MfluxModel[]>;
  by_action?: Record<string, MfluxModel[]>;
  actions?: Record<string, unknown>;
  defaults?: Record<string, string>;
  version_range?: string;
}

export interface MfluxLoraAdapter {
  name: string;
  display_name?: string;
  trigger_tag?: string;
  path: string;
  adapter_type?: "image_lora" | string;
  model_id?: string;
  family?: string;
  base_model?: string;
  generation_loadable?: boolean;
  is_loadable?: boolean;
  updated_at?: string;
  metadata?: Record<string, unknown>;
}

export interface MfluxJob {
  id: string;
  kind?: "mflux";
  state?: string;
  status?: string;
  stage?: string;
  progress?: number;
  created_at?: string;
  updated_at?: string;
  finished_at?: string;
  payload?: Record<string, unknown>;
  model?: MfluxModel;
  result?: Record<string, unknown>;
  result_summary?: Record<string, unknown>;
  dataset_summary?: Record<string, unknown>;
  logs?: string[];
  error?: string;
}

export const getMfluxStatus = () => api.get<MfluxStatus>("/api/mflux/status");
export const getMfluxModels = () => api.get<MfluxModelsResponse>("/api/mflux/models");
export const startMfluxJob = (body: Record<string, unknown>) =>
  api.post<{ success: boolean; job_id?: string; job?: MfluxJob; error?: string }>("/api/mflux/jobs", body);
export const getMfluxJob = (jobId: string) =>
  api.get<{ success: boolean; job?: MfluxJob; error?: string }>(`/api/mflux/jobs/${encodeURIComponent(jobId)}`);
export const getMfluxLoras = () =>
  api.get<{ success: boolean; adapters: MfluxLoraAdapter[] }>("/api/mflux/lora/adapters");
export const uploadMfluxImage = (file: File) => {
  const fd = new FormData();
  fd.append("file", file);
  return api.post<{
    success: boolean;
    id?: string;
    upload_id?: string;
    filename?: string;
    path?: string;
    url?: string;
    error?: string;
  }>("/api/mflux/uploads", fd);
};
export const startMfluxLoraTraining = (body: Record<string, unknown>) =>
  api.post<{ success: boolean; job_id?: string; job?: MfluxJob; error?: string }>("/api/mflux/lora/train", body);
export const attachMfluxArt = (body: Record<string, unknown>) =>
  api.post<{ success: boolean; art?: ArtMetadata; error?: string }>("/api/mflux/art/attach", body);

// ---- MLX video studio ---------------------------------------------------

export interface MlxVideoStatus {
  success: boolean;
  ready: boolean;
  platform?: string;
  arch?: string;
  apple_silicon?: boolean;
  video_env_dir?: string;
  video_python?: string;
  python?: { available?: boolean; ok?: boolean; version?: string; reason?: string };
  mlx_available?: boolean;
  mlx?: { available?: boolean; version?: string; reason?: string };
  mlx_video_available?: boolean;
  mlx_video?: { available?: boolean; version?: string; reason?: string };
  commands?: Record<string, string[]>;
  command_help?: Record<string, {
    available?: boolean;
    help_ok?: boolean;
    command?: string[];
    reason?: string;
    capabilities?: Record<string, boolean | string | number>;
    output_flag?: string;
    help_excerpt?: string;
  }>;
  patch_status?: Record<string, unknown>;
  registered_model_dirs?: MlxVideoModelDir[];
  results_dir?: string;
  uploads_dir?: string;
  lora_dir?: string;
  model_dirs_dir?: string;
  blocking_reason?: string;
}

export interface MlxVideoModel {
  id: string;
  label: string;
  engine: "ltx" | "wan" | string;
  preset?: string;
  pipeline?: string;
  family?: string;
  model_repo?: string;
  requires_model_dir?: boolean;
  default_width?: number;
  default_height?: number;
  default_frames?: number;
  default_fps?: number;
  default_steps?: number;
  guide_scale?: number | string;
  shift?: number | string;
  supports_lora?: boolean;
  capabilities?: string[];
  description?: string;
  disabled?: boolean;
}

export interface MlxVideoModelDir {
  id: string;
  label: string;
  path: string;
  family?: string;
  exists?: boolean;
  config_path?: string;
}

export interface MlxVideoModelsResponse {
  success: boolean;
  status?: MlxVideoStatus;
  models: MlxVideoModel[];
  presets?: Record<string, MlxVideoModel[]>;
  actions?: Record<string, { label?: string; requires_prompt?: boolean; requires_image?: boolean; requires_audio?: boolean }>;
  by_action?: Record<string, MlxVideoModel[]>;
  defaults?: Record<string, string>;
  registered_model_dirs?: MlxVideoModelDir[];
}

export interface MlxVideoLoraAdapter {
  name: string;
  display_name?: string;
  path: string;
  adapter_type?: "video_lora" | string;
  family?: string;
  role?: "shared" | "high" | "low" | string;
  model_id?: string;
  generation_loadable?: boolean;
  is_loadable?: boolean;
  updated_at?: string;
  metadata?: Record<string, unknown>;
}

export interface MlxVideoJob {
  id: string;
  kind?: "mlx-video";
  state?: string;
  status?: string;
  stage?: string;
  progress?: number;
  created_at?: string;
  updated_at?: string;
  finished_at?: string;
  payload?: Record<string, unknown>;
  model?: MlxVideoModel;
  result?: Record<string, unknown>;
  result_summary?: Record<string, unknown>;
  logs?: string[];
  error?: string;
}

export interface MlxVideoAttachment {
  video_id: string;
  source?: "mlx-video" | string;
  target_type?: string;
  target_id?: string;
  result_id?: string;
  url?: string;
  video_url?: string;
  poster_url?: string;
  path?: string;
  prompt?: string;
  model_label?: string;
  attached_at?: string;
}

export const getMlxVideoStatus = () => api.get<MlxVideoStatus>("/api/mlx-video/status");
export const getMlxVideoModels = () => api.get<MlxVideoModelsResponse>("/api/mlx-video/models");
export const getMlxVideoJobs = () =>
  api.get<{ success: boolean; jobs: MlxVideoJob[] }>("/api/mlx-video/jobs");
export const startMlxVideoJob = (body: Record<string, unknown>) =>
  api.post<{ success: boolean; job_id?: string; job?: MlxVideoJob; error?: string }>("/api/mlx-video/jobs", body);
export const getMlxVideoJob = (jobId: string) =>
  api.get<{ success: boolean; job?: MlxVideoJob; error?: string }>(`/api/mlx-video/jobs/${encodeURIComponent(jobId)}`);
export const getMlxVideoLoras = () =>
  api.get<{ success: boolean; adapters: MlxVideoLoraAdapter[] }>("/api/mlx-video/loras");
export const startMlxVideoLoraTraining = (body: Record<string, unknown>) =>
  api.post<{ success: boolean; available?: boolean; error?: string }>("/api/mlx-video/lora/train", body);
export const uploadMlxVideoMedia = (file: File) => {
  const fd = new FormData();
  fd.append("file", file);
  return api.post<{
    success: boolean;
    id?: string;
    upload_id?: string;
    filename?: string;
    path?: string;
    url?: string;
    media_kind?: string;
    error?: string;
  }>("/api/mlx-video/uploads", fd);
};
export const registerMlxVideoModelDir = (body: Record<string, unknown>) =>
  api.post<{ success: boolean; entry?: MlxVideoModelDir; model_dirs?: MlxVideoModelDir[]; error?: string }>("/api/mlx-video/model-dirs", body);
export const attachMlxVideo = (body: Record<string, unknown>) =>
  api.post<{ success: boolean; attachment?: MlxVideoAttachment; error?: string }>("/api/mlx-video/attach", body);
export const getMlxVideoAttachments = (params?: { target_type?: string; target_id?: string }) => {
  const search = new URLSearchParams();
  if (params?.target_type) search.set("target_type", params.target_type);
  if (params?.target_id) search.set("target_id", params.target_id);
  const qs = search.toString();
  return api.get<{ success: boolean; attachments: MlxVideoAttachment[] }>(
    `/api/mlx-video/attachments${qs ? `?${qs}` : ""}`,
  );
};
