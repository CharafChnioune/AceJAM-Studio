import { z } from "zod";
import { DEFAULT_LORA_SCALE } from "@/lib/lora";
import { DEFAULT_AUDIO_BACKEND } from "@/lib/audioBackend";
import { OFFICIAL_AUDIO_FORMAT_VALUES } from "@/lib/aceStepSettings";

/**
 * Zod schemas per wizard step. Kept loose enough that AI-fill can hydrate
 * partial payloads without crashing the form. Strict validation only kicks
 * in at the final Review step.
 */

export const optionalString = z
  .string()
  .optional()
  .transform((v) => (v ?? "").trim() || undefined);

export const optionalNumber = z
  .union([z.number(), z.string()])
  .optional()
  .transform((v) => {
    if (v === undefined || v === null || v === "") return undefined;
    const n = typeof v === "number" ? v : Number(v);
    return Number.isFinite(n) ? n : undefined;
  });

const tagsField = z
  .union([z.string(), z.array(z.string())])
  .optional()
  .transform((v) => {
    if (!v) return "";
    if (Array.isArray(v)) return v.filter(Boolean).join(", ");
    return String(v);
  });

const audioFormatField = z
  .preprocess((value) => (value === "wav16" ? "wav" : value), z.enum(OFFICIAL_AUDIO_FORMAT_VALUES as [string, ...string[]]))
  .default("wav32");

const optionalBoolean = z.boolean().optional();

const optionalStringArray = z
  .union([z.string(), z.array(z.string())])
  .optional()
  .transform((value) => {
    if (Array.isArray(value)) return value.filter(Boolean);
    if (typeof value === "string") {
      return value
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean);
    }
    return undefined;
  });

// ---- Simple ----

export const simpleSchema = z.object({
  simple_description: z.string().min(4, "Beschrijf je idee in minstens een paar woorden."),
  artist_name: optionalString,
  title: optionalString,
  caption: tagsField,
  style_profile: z.string().default("auto"),
  tags: tagsField,
  negative_tags: tagsField,
  lyrics: optionalString,
  instrumental: z.boolean().default(false),
  duration: z.number().min(20).max(600).default(60),
  bpm: optionalNumber,
  key_scale: optionalString,
  time_signature: optionalString,
  vocal_language: z.string().default("en"),
  song_model: z.string().default("acestep-v15-xl-sft"),
  audio_backend: z.enum(["mlx", "mps_torch"]).default(DEFAULT_AUDIO_BACKEND),
  quality_profile: z.enum(["draft", "standard", "chart_master"]).default("standard"),
  seed: optionalNumber,
  use_lora: z.boolean().default(false),
  lora_adapter_path: z.string().default(""),
  lora_adapter_name: z.string().default(""),
  use_lora_trigger: z.boolean().default(false),
  lora_trigger_tag: z.string().default(""),
  lora_scale: z.number().min(0).max(1).default(DEFAULT_LORA_SCALE),
  adapter_model_variant: z.string().default(""),
  adapter_song_model: z.string().default(""),
  auto_song_art: z.boolean().default(false),
  auto_album_art: z.boolean().default(false),
  auto_video_clip: z.boolean().default(false),
  art_prompt: z.string().default(""),
  video_prompt: z.string().default(""),
});

export type SimpleFormValues = z.infer<typeof simpleSchema>;

export const simpleDefaults: SimpleFormValues = {
  simple_description: "",
  artist_name: undefined,
  title: undefined,
  caption: "",
  style_profile: "auto",
  tags: "",
  negative_tags: "",
  lyrics: undefined,
  instrumental: false,
  duration: 60,
  bpm: undefined,
  key_scale: undefined,
  time_signature: undefined,
  vocal_language: "en",
  song_model: "acestep-v15-xl-sft",
  audio_backend: DEFAULT_AUDIO_BACKEND,
  quality_profile: "standard",
  seed: undefined,
  use_lora: false,
  lora_adapter_path: "",
  lora_adapter_name: "",
  use_lora_trigger: false,
  lora_trigger_tag: "",
  lora_scale: DEFAULT_LORA_SCALE,
  adapter_model_variant: "",
  adapter_song_model: "",
  auto_song_art: false,
  auto_album_art: false,
  auto_video_clip: false,
  art_prompt: "",
  video_prompt: "",
};

// ---- Custom ----

export const customSchema = simpleSchema.extend({
  task_type: z.enum(["text2music", "cover", "repaint", "extract", "lego", "complete"]).default(
    "text2music",
  ),
  inference_steps: z.number().int().min(4).max(100).default(50),
  guidance_scale: z.number().min(1).max(15).default(7.0),
  shift: z.number().min(0).max(10).default(1.0),
  audio_format: audioFormatField,
  batch_size: z.number().int().min(1).max(8).default(1),
  infer_method: z.enum(["ode", "sde"]).optional(),
  sampler_mode: z.enum(["euler", "heun"]).optional(),
  use_adg: optionalBoolean,
  cfg_interval_start: optionalNumber,
  cfg_interval_end: optionalNumber,
  timesteps: optionalString,
  dcw_enabled: optionalBoolean,
  dcw_mode: z.enum(["low", "high", "double", "pix"]).optional(),
  dcw_scaler: optionalNumber,
  dcw_high_scaler: optionalNumber,
  dcw_wavelet: z.enum(["haar", "db4", "sym8", "coif1", "bior2.2"]).optional(),
  retake_seed: optionalString,
  retake_variance: optionalNumber,
  enable_normalization: optionalBoolean,
  normalization_db: optionalNumber,
  fade_in_duration: optionalNumber,
  fade_out_duration: optionalNumber,
  latent_shift: optionalNumber,
  latent_rescale: optionalNumber,
  mp3_bitrate: optionalString,
  mp3_sample_rate: optionalNumber,
  auto_score: optionalBoolean,
  auto_lrc: optionalBoolean,
  return_audio_codes: optionalBoolean,
  save_to_library: optionalBoolean,
  use_tiled_decode: optionalBoolean,
  is_format_caption: optionalBoolean,
  device: z.enum(["auto", "mps", "cuda", "cpu"]).optional(),
  dtype: z.enum(["auto", "float32", "float16", "bfloat16"]).optional(),
  use_flash_attention: z.enum(["auto", "true", "false"]).optional(),
  compile_model: optionalBoolean,
  offload_to_cpu: optionalBoolean,
  offload_dit_to_cpu: optionalBoolean,
  vae_checkpoint: optionalString,
  reference_audio: optionalString,
  src_audio: optionalString,
  audio_codes: optionalString,
  global_caption: optionalString,
  instruction: optionalString,
  audio_cover_strength: optionalNumber,
  cover_noise_strength: optionalNumber,
  repainting_start: optionalNumber,
  repainting_end: optionalNumber,
  chunk_mask_mode: z.enum(["auto", "explicit"]).optional(),
  repaint_mode: z.enum(["balanced", "conservative", "aggressive"]).optional(),
  repaint_strength: optionalNumber,
  repaint_latent_crossfade_frames: optionalNumber,
  repaint_wav_crossfade_sec: optionalNumber,
  flow_edit_morph: optionalBoolean,
  flow_edit_source_caption: optionalString,
  flow_edit_source_lyrics: optionalString,
  flow_edit_n_min: optionalNumber,
  flow_edit_n_max: optionalNumber,
  flow_edit_n_avg: optionalNumber,
  track_name: optionalString,
  track_classes: optionalStringArray,
});

export type CustomFormValues = z.infer<typeof customSchema>;

// ---- Album ----

export const albumTrackSchema = z.object({
  track_number: optionalNumber,
  title: optionalString,
  artist_name: optionalString,
  role: optionalString,
  duration: optionalNumber,
  caption: optionalString,
  tags: tagsField,
  lyrics: optionalString,
  bpm: optionalNumber,
  key_scale: optionalString,
  time_signature: optionalString,
  payload_gate_status: optionalString,
  lyrics_quality: z.record(z.string(), z.unknown()).optional(),
  debug_paths: z.record(z.string(), z.unknown()).optional(),
  lora_adapter_name: optionalString,
  lora_scale: optionalNumber,
  lora_trigger_tag: optionalString,
  lora_trigger_applied: z.boolean().optional(),
}).passthrough();

export const albumSchema = z.object({
  concept: z.string().min(8, "Beschrijf het album-concept iets uitgebreider."),
  album_title: optionalString,
  artist_name: optionalString,
  num_tracks: z.number().int().min(1).max(20).default(7),
  track_duration: z.number().min(30).max(600).default(180),
  duration_mode: z.enum(["ai_per_track", "fixed"]).default("ai_per_track"),
  album_writer_mode: z.enum(["per_track_writer_loop"]).default("per_track_writer_loop"),
  language: z.string().default("en"),
  song_model: z.string().default("acestep-v15-xl-sft"),
  audio_backend: z.enum(["mlx", "mps_torch"]).default(DEFAULT_AUDIO_BACKEND),
  song_model_strategy: z.enum(["single_model_album", "all_models_album"]).default(
    "single_model_album",
  ),
  quality_profile: z.enum(["draft", "standard", "chart_master"]).default("standard"),
  album_mood: optionalString,
  vocal_type: optionalString,
  genre_prompt: optionalString,
  style_profile: z.string().default("auto"),
  custom_tags: tagsField,
  negative_tags: tagsField,
  use_lora: z.boolean().default(false),
  lora_adapter_path: z.string().default(""),
  lora_adapter_name: z.string().default(""),
  use_lora_trigger: z.boolean().default(false),
  lora_trigger_tag: z.string().default(""),
  lora_scale: z.number().min(0).max(1).default(DEFAULT_LORA_SCALE),
  adapter_model_variant: z.string().default(""),
  adapter_song_model: z.string().default(""),
  auto_song_art: z.boolean().default(false),
  auto_album_art: z.boolean().default(false),
  auto_video_clip: z.boolean().default(false),
  art_prompt: z.string().default(""),
  video_prompt: z.string().default(""),
  tracks: z.array(albumTrackSchema).default([]),
});

export type AlbumFormValues = z.infer<typeof albumSchema>;
export type AlbumTrackFormValues = z.infer<typeof albumTrackSchema>;
