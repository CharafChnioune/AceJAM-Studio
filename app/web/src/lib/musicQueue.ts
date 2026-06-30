export const PROMPT_COMPANION_KEYS = [
  "genre_execution_contract",
  "lyric_technique_report",
  "lora_selection_reason",
  "performance_notes",
  "strict_completion_notes",
  "single_art_negative_prompt",
  "video_negative_prompt",
  "payload_gate_status",
  "payload_gate_passed",
  "payload_gate_blocking_issues",
  "payload_quality_gate",
  "rap_quality_report",
  "rap_rewrite_status",
  "rap_blocking_issues",
  "rap_strengths",
  "rap_revision_focus",
] as const;

export type PromptCompanionKey = (typeof PROMPT_COMPANION_KEYS)[number];

export interface QueueSummaryItem {
  title: string;
  subtitle?: string;
  detail?: string;
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

export function textOrEmpty(value: unknown): string {
  return typeof value === "string" ? value.trim() : value == null ? "" : String(value).trim();
}

export function extractPromptCompanion(
  payload: Record<string, unknown> | undefined | null,
): Record<string, unknown> {
  if (!payload) return {};
  const companion: Record<string, unknown> = {};
  for (const key of PROMPT_COMPANION_KEYS) {
    if (payload[key] !== undefined) companion[key] = payload[key];
  }
  const visuals = isRecord(payload.visuals) ? payload.visuals : {};
  if (visuals.single_art_negative_prompt !== undefined) {
    companion.single_art_negative_prompt = visuals.single_art_negative_prompt;
  }
  if (visuals.video_negative_prompt !== undefined) {
    companion.video_negative_prompt = visuals.video_negative_prompt;
  }
  return companion;
}

export function mergePayloadWithCompanion(
  payload: Record<string, unknown>,
  companion: Record<string, unknown> | undefined | null,
): Record<string, unknown> {
  if (!companion || Object.keys(companion).length === 0) return { ...payload };
  const merged = { ...payload, ...companion };
  const visuals = isRecord(payload.visuals) ? { ...payload.visuals } : {};
  if (companion.single_art_negative_prompt !== undefined) {
    visuals.single_art_negative_prompt = companion.single_art_negative_prompt;
  }
  if (companion.video_negative_prompt !== undefined) {
    visuals.video_negative_prompt = companion.video_negative_prompt;
  }
  if (Object.keys(visuals).length > 0) merged.visuals = visuals;
  return merged;
}

export function stripPromptCompanion(payload: Record<string, unknown>): Record<string, unknown> {
  const stripped = { ...payload };
  for (const key of PROMPT_COMPANION_KEYS) {
    delete stripped[key];
  }
  const visuals = isRecord(stripped.visuals) ? { ...stripped.visuals } : null;
  if (visuals) {
    delete visuals.single_art_negative_prompt;
    delete visuals.video_negative_prompt;
    if (Object.keys(visuals).length === 0) {
      delete stripped.visuals;
    } else {
      stripped.visuals = visuals;
    }
  }
  return stripped;
}

export function summarizeQueueEntry(
  payload: Record<string, unknown>,
  fallbackLabel: string,
): QueueSummaryItem {
  const title =
    textOrEmpty(payload.title) ||
    textOrEmpty(payload.album_title) ||
    textOrEmpty(payload.sweep_title) ||
    fallbackLabel;
  const subtitle =
    textOrEmpty(payload.artist_name) ||
    textOrEmpty(payload.song_model) ||
    textOrEmpty(payload.concept) ||
    undefined;
  const detail =
    textOrEmpty(payload.lora_adapter_name) ||
    textOrEmpty(payload.lora_trigger_tag) ||
    textOrEmpty(payload.vocal_language) ||
    undefined;
  return { title, subtitle, detail };
}
