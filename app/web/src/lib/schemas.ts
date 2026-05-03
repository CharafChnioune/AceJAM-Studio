import { z } from "zod";

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

// ---- Simple ----

export const simpleSchema = z.object({
  simple_description: z.string().min(4, "Beschrijf je idee in minstens een paar woorden."),
  artist_name: optionalString,
  title: optionalString,
  caption: tagsField,
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
  quality_profile: z.enum(["draft", "standard", "chart_master"]).default("standard"),
  seed: optionalNumber,
});

export type SimpleFormValues = z.infer<typeof simpleSchema>;

export const simpleDefaults: SimpleFormValues = {
  simple_description: "",
  artist_name: undefined,
  title: undefined,
  caption: "",
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
  quality_profile: "standard",
  seed: undefined,
};

// ---- Custom ----

export const customSchema = simpleSchema.extend({
  task_type: z.enum(["text2music", "cover", "repaint", "extract", "lego", "complete"]).default(
    "text2music",
  ),
  inference_steps: z.number().int().min(4).max(100).default(32),
  guidance_scale: z.number().min(1).max(15).default(8.0),
  shift: z.number().min(0).max(10).default(3.0),
  audio_format: z.enum(["wav32", "wav16", "mp3", "flac"]).default("wav32"),
  batch_size: z.number().int().min(1).max(8).default(1),
});

export type CustomFormValues = z.infer<typeof customSchema>;

// ---- Album ----

export const albumSchema = z.object({
  concept: z.string().min(8, "Beschrijf het album-concept iets uitgebreider."),
  album_title: optionalString,
  artist_name: optionalString,
  num_tracks: z.number().int().min(1).max(20).default(7),
  track_duration: z.number().min(30).max(600).default(180),
  language: z.string().default("en"),
  song_model: z.string().default("acestep-v15-xl-sft"),
  song_model_strategy: z.enum(["single_model_album", "all_models_album"]).default(
    "single_model_album",
  ),
  quality_profile: z.enum(["draft", "standard", "chart_master"]).default("standard"),
  album_mood: optionalString,
  vocal_type: optionalString,
  genre_prompt: optionalString,
  custom_tags: tagsField,
  negative_tags: tagsField,
});

export type AlbumFormValues = z.infer<typeof albumSchema>;

// ---- Track-art prompt builder ----

export function defaultTrackArtPrompt(meta: {
  title?: string;
  artist_name?: string;
  caption?: string;
  tags?: string;
}): string {
  const bits = [
    meta.title && `cover artwork for the track "${meta.title}"`,
    meta.artist_name && `by ${meta.artist_name}`,
    meta.caption && `--- vibe: ${meta.caption}`,
    meta.tags && `--- tags: ${meta.tags}`,
    "Stylized, high-contrast, square 1:1 album artwork, no text, no watermark.",
  ].filter(Boolean);
  return bits.join(" ");
}
