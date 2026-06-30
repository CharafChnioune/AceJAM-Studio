import * as React from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { AlertTriangle, Clock3, Loader2, Sparkles, Wand2 } from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  api,
  chatModelDetails,
  embeddingModelDetails,
  getAlbumPlanJob,
  getLLMCatalog,
  getPromptAssistantPrompts,
  promptAssistantRun,
  validatePayload,
  PROVIDER_LABEL,
  startAlbumPlanJob,
  type LLMProvider,
  type WizardMode,
} from "@/lib/api";
import { useSettingsStore } from "@/store/settings";
import { normalizePasteBlocks, normalizeWarnings, useWizardStore } from "@/store/wizard";
import {
  extractPromptCompanion,
  mergePayloadWithCompanion,
} from "@/lib/musicQueue";
import { toast } from "@/components/ui/sonner";

interface ManualApplyResult {
  payload: Record<string, unknown>;
  companion?: Record<string, unknown>;
  queue?: Record<string, unknown>[];
  wrapperKey?: "songs" | "albums" | "sweeps" | "tracks" | "items";
}

interface AIPromptStepProps {
  mode: WizardMode;
  placeholder: string;
  examples?: string[];
  currentPayload?: Record<string, unknown>;
  onHydrated?: (payload: Record<string, unknown>) => void;
  onManualApply?: (result: ManualApplyResult) => void;
  onPendingChange?: (pending: boolean) => void;
}

function modelKey(provider: LLMProvider, name: string): string {
  return `${provider}:${name}`;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

function asText(value: unknown): string {
  return typeof value === "string" ? value : value == null ? "" : String(value);
}

function asNumber(value: unknown, fallback = 0): number {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function firstPresent(...values: unknown[]) {
  for (const value of values) {
    if (value === undefined || value === null) continue;
    if (typeof value === "string" && value.trim() === "") continue;
    return value;
  }
  return undefined;
}

function normalizeQualityProfile(value: unknown): string {
  const normalized = asText(value).trim().toLowerCase().replace(/[-\s]+/g, "_");
  if (["draft", "low", "laag", "fast", "preview", "preview_fast"].includes(normalized)) return "draft";
  if (["standard", "medium", "middle", "middel", "balanced", "balanced_pro"].includes(normalized)) return "standard";
  if (["max_quality", "max", "high", "hoog", "best", "best_quality", "chart_master"].includes(normalized)) return "chart_master";
  return "chart_master";
}

function normalizeSongModel(value: unknown): string {
  const text = asText(value).trim();
  return text || "acestep-v15-xl-sft";
}

function normalizeTimeSignature(value: unknown): string | undefined {
  const text = asText(value).trim();
  if (!text) return undefined;
  if (text.toLowerCase() === "auto") return undefined;
  const match = text.match(/^([2346])(?:\/4|\/8)?$/);
  return match?.[1];
}

function normalizeMusicHydrationPayload(
  payload: Record<string, unknown>,
  mode: WizardMode,
): Record<string, unknown> {
  if (!["simple", "custom", "song", "news", "cover", "repaint", "extract", "lego", "complete"].includes(mode)) {
    return payload;
  }

  const visuals = asRecord(payload.visuals);
  const metadata = asRecord(payload.metadata);
  const generation = asRecord(payload.generation_settings ?? payload.generation);
  const metadataLocks = asRecord(payload.metadata_locks);

  const normalized: Record<string, unknown> = { ...payload };
  const title = asText(firstPresent(payload.title, payload.track_name));
  const artistName = asText(firstPresent(payload.artist_name, payload.artist));
  const caption = asText(firstPresent(payload.caption, payload.ace_caption, payload.tags));
  const tags = asText(firstPresent(payload.tags, payload.caption, payload.ace_caption));
  const negativeTags = asText(firstPresent(payload.negative_tags, payload.negative_control));
  const lyrics = asText(firstPresent(payload.lyrics));
  const keyScale = asText(firstPresent(payload.key_scale, payload.keyscale, metadata.key_scale, metadata.keyscale));
  const timeSignature = normalizeTimeSignature(firstPresent(payload.time_signature, payload.timesignature, metadata.time_signature, metadata.timesignature));
  const duration = firstPresent(payload.duration, payload.audio_duration, metadata.duration);
  const batchSize = firstPresent(payload.batch_size, payload.variant_count, generation.batch_size);
  const guidance = firstPresent(payload.guidance_scale, generation.guidance_scale);
  const inferenceSteps = firstPresent(payload.inference_steps, generation.inference_steps);
  const shift = firstPresent(payload.shift, generation.shift);
  const seed = firstPresent(payload.seed, generation.seed);
  const qualityProfile = normalizeQualityProfile(firstPresent(payload.quality_profile, generation.quality_profile));
  const loraPath = asText(firstPresent(payload.lora_adapter_path));
  const loraName = asText(firstPresent(payload.lora_adapter_name, payload.lora_name));
  const loraTrigger = asText(firstPresent(payload.lora_trigger_tag));
  const taskType = asText(firstPresent(payload.task_type, payload.workflow_mode));

  if (title) normalized.title = title;
  if (artistName) normalized.artist_name = artistName;
  if (caption) normalized.caption = caption;
  if (tags) normalized.tags = tags;
  if (negativeTags) normalized.negative_tags = negativeTags;
  if (lyrics) normalized.lyrics = lyrics;
  normalized.instrumental =
    typeof payload.instrumental === "boolean"
      ? payload.instrumental
      : lyrics.trim() === "[Instrumental]";
  normalized.song_model = normalizeSongModel(firstPresent(payload.song_model, generation.model));
  normalized.quality_profile = qualityProfile;
  if (keyScale && keyScale.toLowerCase() !== "auto") normalized.key_scale = keyScale;
  if (timeSignature) normalized.time_signature = timeSignature;
  if (duration !== undefined) normalized.duration = asNumber(duration, 0) || undefined;
  if (batchSize !== undefined) normalized.batch_size = asNumber(batchSize, 1);
  if (guidance !== undefined) normalized.guidance_scale = asNumber(guidance, 0) || undefined;
  if (inferenceSteps !== undefined) normalized.inference_steps = asNumber(inferenceSteps, 0) || undefined;
  if (shift !== undefined) normalized.shift = asNumber(shift, 0) || undefined;
  if (seed !== undefined) normalized.seed = asNumber(seed, 0) || undefined;
  if (taskType) normalized.task_type = taskType;
  if (payload.audio_duration !== undefined && normalized.duration === undefined) {
    normalized.duration = asNumber(payload.audio_duration, 0) || undefined;
  }
  if (metadataLocks.vocal_language === true && payload.vocal_language) {
    normalized.vocal_language = payload.vocal_language;
  }
  if (payload.vocal_language) normalized.vocal_language = payload.vocal_language;
  if (payload.bpm !== undefined && payload.bpm !== null && `${payload.bpm}`.trim().toLowerCase() !== "auto") {
    normalized.bpm = asNumber(payload.bpm, 0) || undefined;
  }
  if (payload.audio_backend) normalized.audio_backend = payload.audio_backend;
  if (payload.audio_format) normalized.audio_format = payload.audio_format;
  if (payload.infer_method) normalized.infer_method = payload.infer_method;
  if (payload.use_mlx_dit !== undefined) normalized.use_mlx_dit = payload.use_mlx_dit;
  if (loraPath) normalized.lora_adapter_path = loraPath;
  if (loraName) normalized.lora_adapter_name = loraName;
  if (loraTrigger) normalized.lora_trigger_tag = loraTrigger;
  if (payload.use_lora !== undefined) normalized.use_lora = payload.use_lora;
  if (payload.use_lora_trigger !== undefined) normalized.use_lora_trigger = payload.use_lora_trigger;
  if (payload.lora_scale !== undefined) normalized.lora_scale = payload.lora_scale;
  if (payload.lora_trigger_tags !== undefined) normalized.lora_trigger_tags = payload.lora_trigger_tags;
  if (payload.lora_adapters !== undefined) normalized.lora_adapters = payload.lora_adapters;
  if (payload.adapter_model_variant !== undefined) normalized.adapter_model_variant = payload.adapter_model_variant;
  if (payload.adapter_song_model !== undefined) normalized.adapter_song_model = payload.adapter_song_model;
  if (payload.genre_execution_contract !== undefined) normalized.genre_execution_contract = payload.genre_execution_contract;
  if (payload.lyric_technique_report !== undefined) normalized.lyric_technique_report = payload.lyric_technique_report;
  if (payload.lora_selection_reason !== undefined) normalized.lora_selection_reason = payload.lora_selection_reason;
  if (payload.performance_notes !== undefined) normalized.performance_notes = payload.performance_notes;
  if (payload.strict_completion_notes !== undefined) normalized.strict_completion_notes = payload.strict_completion_notes;
  if (payload.payload_gate_status !== undefined) normalized.payload_gate_status = payload.payload_gate_status;
  if (payload.payload_gate_passed !== undefined) normalized.payload_gate_passed = payload.payload_gate_passed;
  if (payload.payload_gate_blocking_issues !== undefined) normalized.payload_gate_blocking_issues = payload.payload_gate_blocking_issues;
  if (payload.payload_quality_gate !== undefined) normalized.payload_quality_gate = payload.payload_quality_gate;
  if (payload.rap_quality_report !== undefined) normalized.rap_quality_report = payload.rap_quality_report;
  if (payload.rap_rewrite_status !== undefined) normalized.rap_rewrite_status = payload.rap_rewrite_status;
  if (payload.rap_blocking_issues !== undefined) normalized.rap_blocking_issues = payload.rap_blocking_issues;
  if (payload.rap_strengths !== undefined) normalized.rap_strengths = payload.rap_strengths;
  if (payload.rap_revision_focus !== undefined) normalized.rap_revision_focus = payload.rap_revision_focus;
  if (visuals.single_art_prompt !== undefined) normalized.art_prompt = visuals.single_art_prompt;
  if (visuals.single_art_negative_prompt !== undefined) {
    normalized.single_art_negative_prompt = visuals.single_art_negative_prompt;
  }
  if (visuals.video_prompt !== undefined) normalized.video_prompt = visuals.video_prompt;
  if (visuals.video_negative_prompt !== undefined) {
    normalized.video_negative_prompt = visuals.video_negative_prompt;
  }
  if (payload.art_prompt !== undefined) normalized.art_prompt = payload.art_prompt;
  if (payload.video_prompt !== undefined) normalized.video_prompt = payload.video_prompt;
  if (payload.single_art_negative_prompt !== undefined) {
    normalized.single_art_negative_prompt = payload.single_art_negative_prompt;
  }
  if (payload.video_negative_prompt !== undefined) {
    normalized.video_negative_prompt = payload.video_negative_prompt;
  }
  return normalized;
}

function stableJson(value: unknown): string {
  try {
    return JSON.stringify(value && typeof value === "object" ? value : {}, null, 2);
  } catch {
    return "{}";
  }
}

function stripJsonFence(value: string): string {
  let text = value.trim();
  if (text.startsWith("```")) {
    text = text.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/m, "").trim();
  }
  return text;
}

function formatJsonParseError(error: unknown): string {
  if (error instanceof SyntaxError) {
    const match = error.message.match(/line\s+(\d+)\s+column\s+(\d+)/i);
    if (match) {
      return `JSON parse fout op regel ${match[1]}, kolom ${match[2]}: ${error.message}`;
    }
    return `JSON parse fout: ${error.message}`;
  }
  if (error instanceof Error && error.message) return error.message;
  return "JSON kon niet worden gelezen.";
}

function balancedJsonText(value: string): string {
  const text = value.trim();
  const start = text.search(/[\[{]/);
  if (start < 0) return text;
  const opener = text[start];
  const closer = opener === "{" ? "}" : "]";
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let index = start; index < text.length; index += 1) {
    const char = text[index];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === "\"") {
        inString = false;
      }
      continue;
    }
    if (char === "\"") {
      inString = true;
      continue;
    }
    if (char === opener) {
      depth += 1;
    } else if (char === closer) {
      depth -= 1;
      if (depth === 0) return text.slice(start, index + 1);
    }
  }
  return text.slice(start);
}

function normalizeManualEntry(record: Record<string, unknown>, mode: WizardMode): Record<string, unknown> {
  const normalized = normalizeMusicHydrationPayload(record, mode);
  return mergePayloadWithCompanion(normalized, extractPromptCompanion(record));
}

function parseManualPastePayload(raw: string, mode: WizardMode): ManualApplyResult {
  let text = stripJsonFence(raw);
  if (!text) return { payload: {} };
  const markerIndex = text.indexOf("ACEJAM_PAYLOAD_JSON");
  if (markerIndex >= 0) {
    text = text.slice(markerIndex + "ACEJAM_PAYLOAD_JSON".length).trim();
  }
  text = balancedJsonText(text);
  const parsed = JSON.parse(text) as unknown;
  let payload: unknown = parsed;
  if (payload && typeof payload === "object" && !Array.isArray(payload)) {
    const record = payload as Record<string, unknown>;
    if (record.payload && typeof record.payload === "object" && !Array.isArray(record.payload)) {
      payload = record.payload;
    } else if (
      record.ACEJAM_PAYLOAD_JSON &&
      typeof record.ACEJAM_PAYLOAD_JSON === "object" &&
      !Array.isArray(record.ACEJAM_PAYLOAD_JSON)
    ) {
      const nested = record.ACEJAM_PAYLOAD_JSON as Record<string, unknown>;
      payload =
        nested.payload && typeof nested.payload === "object" && !Array.isArray(nested.payload)
          ? nested.payload
          : nested;
    }
  }
  if (Array.isArray(payload)) {
    const queue = payload
      .filter((item): item is Record<string, unknown> => Boolean(item && typeof item === "object" && !Array.isArray(item)))
      .map((item) => normalizeManualEntry(item, mode));
    const [current, ...rest] = queue;
    return {
      payload: current ?? {},
      companion: extractPromptCompanion(current),
      queue: rest,
      wrapperKey: mode === "album" ? "albums" : "songs",
    };
  }
  if (!payload || typeof payload !== "object") {
    throw new Error("JSON root must be an object.");
  }
  const record = payload as Record<string, unknown>;
  const summary = asRecord(record.summary);
  const aceStepRequest = asRecord(record.ace_step_request ?? record.official_request);
  const aceStepParams = asRecord(aceStepRequest.params);
  if (Object.keys(aceStepParams).length > 0) {
    const normalized: Record<string, unknown> = {
      ...aceStepParams,
      song_model: asText(aceStepRequest.song_model) || asText(record.song_model) || asText(summary.song_model),
      title: asText(aceStepParams.title) || asText(summary.title),
      task_type: asText(aceStepParams.task_type) || asText(summary.task_type),
      lora_adapter_name: asText(aceStepRequest.lora_adapter_name),
      lora_adapter_path: asText(aceStepRequest.lora_adapter_path),
      lora_scale: aceStepRequest.lora_scale,
      adapter_model_variant: asText(aceStepRequest.adapter_model_variant),
      key_scale: asText(aceStepParams.key_scale) || asText(aceStepParams.keyscale),
      duration: asNumber(aceStepParams.duration, asNumber(aceStepParams.audio_duration, 0)) || undefined,
      audio_duration:
        asNumber(aceStepParams.audio_duration, asNumber(aceStepParams.duration, 0)) || undefined,
    };
    if (aceStepRequest.use_lora != null) normalized.use_lora = aceStepRequest.use_lora;
    if (mode === "complete" || mode === "lego" || mode === "extract") {
      normalized.track_names = Array.isArray(aceStepParams.track_names) ? aceStepParams.track_names : [];
    }
    const merged = normalizeManualEntry(normalized, mode);
    return { payload: merged, companion: extractPromptCompanion(merged) };
  }

  const wrapperMap: Array<["songs" | "albums" | "sweeps" | "items" | "tracks", unknown]> = [
    ["songs", record.songs],
    ["albums", record.albums],
    ["sweeps", record.sweeps],
    ["items", record.items],
    ["tracks", mode === "album" ? record.tracks : undefined],
  ];
  for (const [wrapperKey, value] of wrapperMap) {
    if (!Array.isArray(value)) continue;
    const queue = value
      .filter((item): item is Record<string, unknown> => Boolean(item && typeof item === "object" && !Array.isArray(item)))
      .map((item) => normalizeManualEntry(item, mode));
    if (queue.length > 0) {
      const [current, ...rest] = queue;
      return {
        payload: current,
        companion: extractPromptCompanion(current),
        queue: rest,
        wrapperKey,
      };
    }
  }
  if (mode === "album" && Array.isArray(record.tracks)) {
    const merged = { ...record, tracks: record.tracks };
    const payloadWithCompanion = mergePayloadWithCompanion(merged, extractPromptCompanion(record));
    return {
      payload: payloadWithCompanion,
      companion: extractPromptCompanion(payloadWithCompanion),
      queue: [payloadWithCompanion],
      wrapperKey: "albums",
    };
  }
  const payloadWithCompanion = normalizeManualEntry(record, mode);
  return { payload: payloadWithCompanion, companion: extractPromptCompanion(payloadWithCompanion) };
}

const ALBUM_CONTRACT_NEGATIVE_TAGS =
  "low quality, muddy mix, distorted vocals, off-key vocals, clipped audio, noisy artifacts";

function styleProfileFromText(value: unknown): string {
  const text = asText(value).toLowerCase();
  if (/\b(rap|hip[-\s]?hop|trap|drill|boom[-\s]?bap|g[-\s]?funk|west coast)\b/.test(text)) return "rap";
  if (/\b(r&b|rnb|soul)\b/.test(text)) return "soul";
  if (/\b(edm|dance|house|techno|club|trance|bounce)\b/.test(text)) return "edm";
  if (/\b(rock|punk|metal|guitar)\b/.test(text)) return "rock";
  if (/\b(country|americana|folk)\b/.test(text)) return "country";
  if (/\b(cinematic|score|orchestral|trailer)\b/.test(text)) return "cinematic";
  return "pop";
}

function contractCaptionTags(style: string, profile: string): string {
  const base =
    profile === "rap"
      ? "rap, hip hop, clear rap vocal, rhythmic spoken flow, punchy drums, deep bass, crisp studio mix"
      : profile === "soul"
        ? "soul, R&B, warm vocal, live bass, tight drums, rich harmonies, polished studio mix"
        : profile === "edm"
          ? "dance, club groove, driving kick, deep bass, energetic vocal, polished electronic mix"
          : profile === "cinematic"
            ? "cinematic, orchestral texture, dramatic percussion, wide stereo mix, emotional vocal"
            : "pop, catchy vocal hook, polished drums, warm bass, radio-ready mix";
  return [style, base].filter(Boolean).join(", ");
}

function albumInitialPayloadFromJob(
  job: Record<string, unknown>,
  fallbackPayload: Record<string, unknown>,
): Record<string, unknown> {
  const contract = asRecord(job.input_contract ?? job.user_album_contract);
  const rawTracks = Array.isArray(contract.tracks) ? contract.tracks.map(asRecord) : [];
  if (!rawTracks.length && !asText(contract.album_title)) return {};
  const tracks = rawTracks.map((track, index) => {
    const style = asText(track.style);
    const profile = styleProfileFromText(`${style} ${track.narrative ?? ""} ${track.source_excerpt ?? ""}`);
    const title = asText(track.locked_title) || asText(track.title) || `Track ${index + 1}`;
    const requiredPhrases = Array.isArray(track.required_phrases)
      ? track.required_phrases.map(String).filter(Boolean)
      : [];
    return {
      track_number: asNumber(track.track_number, index + 1),
      title,
      locked_title: title,
      source_title: asText(track.source_title) || title,
      role: index === 0 ? "opener" : index === rawTracks.length - 1 ? "closer" : "full_song",
      duration: asNumber(track.duration, asNumber(fallbackPayload.track_duration, 180)) || 180,
      bpm: asNumber(track.bpm, 0) || undefined,
      key_scale: asText(track.key_scale),
      style,
      style_profile: profile,
      genre_profile: style || profile,
      genre_direction: style || profile,
      caption_tags: contractCaptionTags(style, profile),
      album_tags: asText(fallbackPayload.genre_prompt) || style || profile,
      negative_tags: ALBUM_CONTRACT_NEGATIVE_TAGS,
      narrative: asText(track.narrative) || asText(track.source_excerpt),
      description: asText(track.narrative) || asText(track.source_excerpt) || style,
      hook_promise: requiredPhrases.slice(0, 6).join("\n"),
      required_phrases: requiredPhrases,
      lyrics: asText(track.required_lyrics),
      input_contract_applied: true,
      payload_gate_status: "contract_pending_crewai",
    };
  });
  return {
    ...fallbackPayload,
    album_title: asText(contract.album_title) || asText(fallbackPayload.album_title),
    concept: asText(contract.concept) || asText(fallbackPayload.concept),
    num_tracks: asNumber(contract.track_count, tracks.length || asNumber(fallbackPayload.num_tracks, 0)),
    tracks,
    input_contract: contract,
    input_contract_applied: true,
  };
}

export function AIPromptStep({
  mode,
  placeholder,
  examples,
  currentPayload,
  onHydrated,
  onManualApply,
  onPendingChange,
}: AIPromptStepProps) {
  const prompt = useWizardStore((s) => s.prompts[mode]) ?? "";
  const promptPreset = useWizardStore((s) => s.promptPresets[mode]) ?? "";
  const warnings = normalizeWarnings(useWizardStore((s) => s.warnings[mode]));
  const storedPayload = useWizardStore((s) => s.payloads[mode]);
  const storedCompanion = useWizardStore((s) => s.companions[mode]);
  const storedPasteBlocks = useWizardStore((s) => s.pasteBlocks[mode]);
  const pasteBlocks = normalizePasteBlocks(storedPasteBlocks);
  const setPrompt = useWizardStore((s) => s.setPrompt);
  const setPromptPreset = useWizardStore((s) => s.setPromptPreset);
  const setPasteBlocks = useWizardStore((s) => s.setPasteBlocks);
  const setHydration = useWizardStore((s) => s.setHydration);

  const plannerProvider = useSettingsStore((s) => s.plannerProvider);
  const plannerModel = useSettingsStore((s) => s.plannerModel);
  const setPlanner = useSettingsStore((s) => s.setPlanner);
  const embeddingProvider = useSettingsStore((s) => s.embeddingProvider);
  const embeddingModel = useSettingsStore((s) => s.embeddingModel);
  const setEmbedding = useSettingsStore((s) => s.setEmbedding);

  const catalogQuery = useQuery({
    queryKey: ["llm-catalog"],
    queryFn: getLLMCatalog,
    staleTime: 30_000,
  });

  const promptCatalogQuery = useQuery({
    queryKey: ["prompt-assistant-prompts"],
    queryFn: getPromptAssistantPrompts,
    staleTime: 300_000,
  });

  const allChatModels = React.useMemo(
    () => chatModelDetails(catalogQuery.data),
    [catalogQuery.data],
  );

  const allEmbeddingModels = React.useMemo(
    () =>
      embeddingModelDetails(catalogQuery.data).filter(
        (m) => m.provider === "ollama" || m.provider === "lmstudio",
      ),
    [catalogQuery.data],
  );

  // Group by provider for the dropdowns
  const groupedChatModels = React.useMemo(() => {
    const map = new Map<LLMProvider, typeof allChatModels>();
    for (const m of allChatModels) {
      const arr = map.get(m.provider) ?? [];
      arr.push(m);
      map.set(m.provider, arr);
    }
    return map;
  }, [allChatModels]);

  const groupedEmbeddingModels = React.useMemo(() => {
    const map = new Map<LLMProvider, typeof allEmbeddingModels>();
    for (const m of allEmbeddingModels) {
      const arr = map.get(m.provider) ?? [];
      arr.push(m);
      map.set(m.provider, arr);
    }
    return map;
  }, [allEmbeddingModels]);

  const availablePromptPresets = React.useMemo(
    () =>
      (promptCatalogQuery.data?.prompts.find((item) => item.mode === mode)?.presets ?? []).filter(
        (preset) => preset.available,
      ),
    [mode, promptCatalogQuery.data],
  );

  // Auto-pick a sensible default planner from catalog.settings or first chat model
  React.useEffect(() => {
    if (plannerModel || !catalogQuery.data) return;
    const settings = catalogQuery.data.settings;
    const preferred = settings?.chat_model;
    const preferredProvider = (settings?.provider ?? "ollama") as LLMProvider;
    if (preferred && allChatModels.some((m) => m.provider === preferredProvider && m.name === preferred)) {
      setPlanner(preferredProvider, preferred);
      return;
    }
    if (allChatModels.length > 0) {
      const first = allChatModels[0];
      setPlanner(first.provider, first.name);
    }
  }, [catalogQuery.data, allChatModels, plannerModel, setPlanner]);

  React.useEffect(() => {
    if (embeddingModel || !catalogQuery.data) return;
    const settings = catalogQuery.data.settings;
    const preferred = settings?.embedding_model;
    const preferredProvider = (settings?.embedding_provider ?? "ollama") as LLMProvider;
    if (
      preferred &&
      allEmbeddingModels.some((m) => m.provider === preferredProvider && m.name === preferred)
    ) {
      setEmbedding(preferredProvider, preferred);
      return;
    }
    if (allEmbeddingModels.length > 0) {
      const first = allEmbeddingModels[0];
      setEmbedding(first.provider, first.name);
    }
  }, [allEmbeddingModels, catalogQuery.data, embeddingModel, setEmbedding]);

  React.useEffect(() => {
    if (!promptPreset) return;
    if (availablePromptPresets.some((preset) => preset.id === promptPreset)) return;
    setPromptPreset(mode, undefined);
  }, [availablePromptPresets, mode, promptPreset, setPromptPreset]);

  const currentKey = plannerModel ? modelKey(plannerProvider, plannerModel) : "";
  const embeddingKey = embeddingModel ? modelKey(embeddingProvider, embeddingModel) : "";
  const [albumJobId, setAlbumJobId] = React.useState("");
  const [albumJob, setAlbumJob] = React.useState<Record<string, unknown> | null>(null);
  const [manualApplyError, setManualApplyError] = React.useState("");

  const saveLocalSettings = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      api.post<{ success: boolean; error?: string }>("/api/local-llm/settings", body),
    onSuccess: (data) => {
      if (!data.success) toast.error(data.error || "AI Memory-instelling opslaan mislukt");
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const aiFill = useMutation({
    mutationFn: () => {
      const payload = currentPayload ?? {};
      if (mode === "album") {
        return startAlbumPlanJob({
          ...payload,
          concept: prompt.trim() || asText(payload.concept),
          user_prompt: prompt,
          prompt,
          agent_engine: asText(payload.agent_engine) || "crewai_micro",
          album_writer_mode: "per_track_writer_loop",
          planner_lm_provider: plannerProvider,
          planner_model: plannerModel || undefined,
          ollama_model: plannerProvider === "ollama" ? plannerModel || undefined : asText(payload.ollama_model) || undefined,
          embedding_provider: embeddingProvider,
          embedding_lm_provider: embeddingProvider,
          embedding_model: embeddingModel || catalogQuery.data?.settings?.embedding_model || undefined,
        });
      }
      return promptAssistantRun({
        mode,
        user_prompt: prompt,
        prompt_preset: promptPreset || undefined,
        current_payload: currentPayload,
        planner_lm_provider: plannerProvider,
        planner_model: plannerModel || undefined,
        embedding_provider: embeddingProvider,
        embedding_lm_provider: embeddingProvider,
        embedding_model: embeddingModel || catalogQuery.data?.settings?.embedding_model || undefined,
      });
    },
    onMutate: () => {
      if (mode === "album") {
        const totalTracks = asNumber((currentPayload ?? {}).num_tracks, 0);
        setAlbumJobId("");
        setAlbumJob({
          state: "starting",
          stage: "starting",
          status: "Album AI job starten",
          current_task: "Backend job-id ophalen",
          current_agent: "Album planner",
          current_track: 0,
          total_tracks: totalTracks || "?",
          completed_tracks: 0,
          remaining_tracks: totalTracks || "?",
          progress: 1,
          waiting_on_llm: false,
          logs: ["Album AI job wordt gestart via de background planner."],
        });
      }
    },
    onSuccess: (data) => {
      const response = data as {
        success?: boolean;
        error?: string;
        message?: string;
        logs?: string[];
        job_id?: string;
        job?: unknown;
        ollama_pull_started?: boolean;
        ollama_model?: string;
        ollama_pull_job?: unknown;
      };
      if (mode === "album" && response.ollama_pull_started) {
        const pullJob = asRecord(response.ollama_pull_job);
        const model = asText(response.ollama_model) || asText(pullJob.model) || "Ollama model";
        const progress = asNumber(pullJob.progress, 0);
        const status = asText(pullJob.status) || "pulling model";
        const logs = Array.isArray(response.logs)
          ? response.logs.map(String)
          : Array.isArray(pullJob.logs)
            ? pullJob.logs.map(String)
            : [asText(response.message) || `${model} wordt gedownload.`];
        setAlbumJobId("");
        setAlbumJob({
          id: asText(pullJob.id),
          state: asText(pullJob.state) || "pulling",
          stage: "model_download",
          status,
          current_task: `Planner-model downloaden: ${model}`,
          current_agent: "Local LLM install",
          progress,
          waiting_on_llm: false,
          logs,
        });
        toast.info(`${model} wordt eerst gedownload. Start Album AI Fill opnieuw zodra de download klaar is.`);
        return;
      }
      if (!response.success) {
        toast.error(response.error || "AI-fill mislukte");
        return;
      }
      if (mode === "album" && "job_id" in data) {
        const jobId = asText(response.job_id);
        if (!jobId) {
          toast.error("Album AI job kon niet worden gestart.");
          return;
        }
        const initialPayload = albumInitialPayloadFromJob(asRecord(response.job), currentPayload ?? {});
        if (Object.keys(initialPayload).length > 0) {
          setHydration(mode, {
            payload: initialPayload,
            warnings: [
              "Albumcontract alvast ingevuld; CrewAI werkt nu track-voor-track aan verdere verrijking.",
            ],
            validation: null,
            paste_blocks: null,
          });
          onHydrated?.(initialPayload);
        }
        setAlbumJobId(jobId);
        setAlbumJob(asRecord(response.job));
        toast.success("Album AI Fill gestart. Je ziet live welke taak bezig is.");
        return;
      }
      const assistantData = data as {
        payload?: Record<string, unknown>;
        validation?: Record<string, unknown> | null;
        warnings?: string[] | string | null;
        paste_blocks?: Array<{ label?: string; content?: string; text?: string }> | string | null;
      };
      setHydration(mode, {
        payload: assistantData.payload,
        validation: assistantData.validation ?? null,
        warnings: assistantData.warnings,
        paste_blocks: assistantData.paste_blocks,
      });
      if (assistantData.payload) onHydrated?.(assistantData.payload);
      toast.success("AI heeft het wizard-formulier voorgevuld.");
    },
    onError: (err: Error) => toast.error(err.message),
  });

  React.useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      if (!albumJobId) return;
      try {
        const data = await getAlbumPlanJob(albumJobId);
        if (cancelled) return;
        const job = asRecord(data.job);
        setAlbumJob(job);
        const state = asText(job.state).toLowerCase();
        if (state === "succeeded") {
          const result = asRecord(job.result);
          const payload = asRecord(result.payload).tracks ? asRecord(result.payload) : result;
          setHydration(mode, {
            payload,
            validation: null,
            warnings: result.warnings,
            paste_blocks: result.paste_blocks,
          });
          if (Object.keys(payload).length > 0) onHydrated?.(payload);
          setAlbumJobId("");
          toast.success("Album AI heeft het wizard-formulier voorgevuld.");
          return;
        }
        if (state === "failed" || state === "needs_review") {
          const errors = Array.isArray(job.errors) ? job.errors.map(String) : [];
          setAlbumJobId("");
          toast.error(errors[0] || asText(job.error) || "Album AI Fill mislukte");
          return;
        }
      } catch (err) {
        if (!cancelled) toast.error(err instanceof Error ? err.message : "Album jobstatus laden mislukt");
      }
      if (!cancelled) timer = setTimeout(poll, 2000);
    };
    if (albumJobId) {
      timer = setTimeout(poll, 600);
    }
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [albumJobId, mode, onHydrated, setHydration]);

  const albumFillPending = Boolean(albumJobId);
  const pending = aiFill.isPending || albumFillPending;

  React.useEffect(() => {
    // Parity anchor: onPendingChange?.(aiFill.isPending) is now extended with album job polling.
    onPendingChange?.(pending);
  }, [pending, onPendingChange]);

  const onModelChange = (key: string) => {
    if (!key) return;
    const idx = key.indexOf(":");
    if (idx < 0) return;
    const provider = key.slice(0, idx) as LLMProvider;
    const name = key.slice(idx + 1);
    setPlanner(provider, name);
  };

  const onEmbeddingChange = (key: string) => {
    if (!key) return;
    const idx = key.indexOf(":");
    if (idx < 0) return;
    const provider = key.slice(0, idx) as LLMProvider;
    const name = key.slice(idx + 1);
    setEmbedding(provider, name);
    saveLocalSettings.mutate({
      embedding_provider: provider,
      embedding_model: name,
    });
  };

  const isLoading = catalogQuery.isLoading;
  const isEmpty = !isLoading && allChatModels.length === 0;
  const embeddingsEmpty = !isLoading && allEmbeddingModels.length === 0;
  const albumWaiting =
    Boolean(albumJob?.waiting_on_llm) ||
    asText(albumJob?.waiting_on_llm).toLowerCase() === "true";
  const pasteEditorFallback = React.useMemo(
    () =>
      stableJson(
        mergePayloadWithCompanion(currentPayload ?? storedPayload ?? {}, storedCompanion),
      ),
    [currentPayload, storedCompanion, storedPayload],
  );
  const editablePasteBlocks = React.useMemo(
    () =>
      storedPasteBlocks && storedPasteBlocks.length > 0
        ? storedPasteBlocks
        : [
            {
              label: "Huidige wizard JSON",
              content: pasteEditorFallback,
            },
          ],
    [pasteEditorFallback, storedPasteBlocks],
  );
  const pasteEditorValue = editablePasteBlocks[0]?.content ?? pasteEditorFallback;
  const promptPresetValue =
    availablePromptPresets.length > 0 ? promptPreset || "__wizard_default__" : "";

  const updatePasteBlock = (index: number, content: string) => {
    setManualApplyError("");
    const next = editablePasteBlocks.map((block) => ({ ...block }));
    next[index] = {
      label: next[index]?.label || `Paste block ${index + 1}`,
      content,
    };
    setPasteBlocks(mode, next);
  };

  const resetPasteEditor = () => {
    setManualApplyError("");
    setPasteBlocks(mode, [
      {
        label: "Huidige wizard JSON",
        content: pasteEditorFallback,
      },
    ]);
  };

  const applyPasteEditor = async (index = 0) => {
    const block = editablePasteBlocks[index] ?? editablePasteBlocks[0];
    const content = block?.content ?? pasteEditorValue;
    try {
      const applied = parseManualPastePayload(content, mode);
      const manualPayload = applied.payload;
      const validation =
        mode === "album" || mode === "image" || mode === "video" || (applied.queue?.length ?? 0) > 1
          ? null
          : await validatePayload(manualPayload);
      setHydration(mode, {
        payload: manualPayload,
        validation,
        warnings,
        paste_blocks: [
          {
            label: "Huidige wizard JSON",
            content: stableJson(manualPayload),
          },
          ...editablePasteBlocks.slice(1),
        ],
        companion: applied.companion ?? extractPromptCompanion(manualPayload),
        queue: applied.queue,
      });
      onHydrated?.(manualPayload);
      onManualApply?.(applied);
      setManualApplyError("");
      toast.success(
        applied.queue && applied.queue.length > 0
          ? `JSON toegepast: huidig item geladen, ${applied.queue.length} extra item(s) aan queue toegevoegd.`
          : "JSON toegepast op de wizard.",
      );
    } catch (err) {
      const message = formatJsonParseError(err);
      setManualApplyError(message);
      toast.error(message);
    }
  };

  return (
    <div className="space-y-5">
      <div className="space-y-2">
        <Label htmlFor={`${mode}-prompt`} className="text-sm font-medium">
          Wat wil je maken?
        </Label>
        <Textarea
          id={`${mode}-prompt`}
          placeholder={placeholder}
          value={prompt}
          onChange={(e) => setPrompt(mode, e.target.value)}
          rows={6}
          className="text-base leading-relaxed"
        />
        {examples && examples.length > 0 && (
          <div className="flex flex-wrap gap-1.5 pt-1">
            <span className="text-xs text-muted-foreground">Voorbeelden:</span>
            {examples.map((ex) => (
              <button
                key={ex}
                type="button"
                onClick={() => setPrompt(mode, ex)}
                className="rounded-full border border-border/60 px-2.5 py-0.5 text-xs text-muted-foreground transition-colors hover:border-primary/60 hover:text-foreground"
              >
                {ex}
              </button>
            ))}
          </div>
        )}
      </div>

      {availablePromptPresets.length > 0 && (
        <div className="space-y-2">
          <Label className="text-xs uppercase tracking-wider text-muted-foreground">
            Genre Prompt Preset
          </Label>
          <Select
            value={promptPresetValue}
            onValueChange={(value) =>
              setPromptPreset(mode, value === "__wizard_default__" ? undefined : value)
            }
          >
            <SelectTrigger>
              <SelectValue placeholder="Gebruik wizard-standaardprompt" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__wizard_default__">Wizard-standaardprompt</SelectItem>
              {availablePromptPresets.map((preset) => (
                <SelectItem key={preset.id} value={preset.id}>
                  <div className="flex items-center gap-2">
                    <span>{preset.label}</span>
                    {preset.family ? (
                      <span className="text-[10px] text-muted-foreground uppercase tracking-wide">
                        {preset.family}
                      </span>
                    ) : null}
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-[11px] text-muted-foreground">
            Rap-presets forceren extra <code>lyric_technique_report</code> analyse zoals
            line-intent, word hits en ad-libs. De uiteindelijke <code>lyrics</code> blijven
            ACE-Step-schoon zonder inline kleurmarkup of HTML.
          </p>
        </div>
      )}

      <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] lg:items-end">
        <div className="space-y-2">
          <Label className="text-xs uppercase tracking-wider text-muted-foreground">
            Planner model
          </Label>
          {isEmpty ? (
            <div className="rounded-md border border-yellow-500/30 bg-yellow-500/5 p-3 text-xs text-yellow-200">
              Geen chat-modellen gevonden bij Ollama, LM Studio of de officiële
              ACE-Step LM. Open <Link to="/settings" className="underline">Settings</Link> of
              install een Ollama model met <code className="rounded bg-background/40 px-1">ollama pull qwen3:14b</code>.
            </div>
          ) : (
            <Select value={currentKey} onValueChange={onModelChange}>
              <SelectTrigger>
                <SelectValue placeholder={isLoading ? "Catalog laden…" : "Kies een planner-model"} />
              </SelectTrigger>
              <SelectContent>
                {Array.from(groupedChatModels.entries()).map(([provider, list]) => (
                  <SelectGroup key={provider}>
                    <SelectLabel>{PROVIDER_LABEL[provider]}</SelectLabel>
                    {list.map((m) => {
                      const dropdownLabel =
                        m.profile?.dropdown_label || m.display_name || m.name;
                      return (
                        <SelectItem key={m.key} value={modelKey(m.provider, m.name)}>
                          <div className="flex items-center gap-2">
                            <span>{dropdownLabel}</span>
                            {m.size_gb && (
                              <span className="text-[10px] text-muted-foreground">
                                {m.size_gb.toFixed(1)} GB
                              </span>
                            )}
                          </div>
                        </SelectItem>
                      );
                    })}
                  </SelectGroup>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>
        <div className="space-y-2">
          <Label className="text-xs uppercase tracking-wider text-muted-foreground">
            AI Memory / RAG embedding
          </Label>
          {embeddingsEmpty ? (
            <div className="rounded-md border border-yellow-500/30 bg-yellow-500/5 p-3 text-xs text-yellow-200">
              Geen embedding-modellen gevonden bij Ollama of LM Studio. Open{" "}
              <Link to="/settings" className="underline">Settings</Link>{" "}
              en kies of pull een embedding-model voor CrewAI memory/RAG.
            </div>
          ) : (
            <Select value={embeddingKey} onValueChange={onEmbeddingChange}>
              <SelectTrigger>
                <SelectValue placeholder={isLoading ? "Catalog laden…" : "Kies embedding-model"} />
              </SelectTrigger>
              <SelectContent>
                {Array.from(groupedEmbeddingModels.entries()).map(([provider, list]) => (
                  <SelectGroup key={provider}>
                    <SelectLabel>{PROVIDER_LABEL[provider]}</SelectLabel>
                    {list.map((m) => {
                      const dropdownLabel =
                        m.profile?.dropdown_label || m.display_name || m.name;
                      return (
                        <SelectItem key={m.key} value={modelKey(m.provider, m.name)}>
                          <div className="flex items-center gap-2">
                            <span>{dropdownLabel}</span>
                            {m.size_gb && (
                              <span className="text-[10px] text-muted-foreground">
                                {m.size_gb.toFixed(1)} GB
                              </span>
                            )}
                          </div>
                        </SelectItem>
                      );
                    })}
                  </SelectGroup>
                ))}
              </SelectContent>
            </Select>
          )}
          <p className="text-[11px] text-muted-foreground">
            Alleen voor AI/CrewAI memory en RAG. ACE-Step audio text encoder blijft Qwen3-Embedding-0.6B.
          </p>
        </div>
        <Button
          size="lg"
          onClick={() => aiFill.mutate()}
          disabled={!prompt.trim() || !plannerModel || pending}
          className="gap-2"
        >
          {pending ? <Loader2 className="size-4 animate-spin" /> : <Sparkles className="size-4" />}
          {pending ? (mode === "album" ? "Album AI loopt…" : "AI denkt na…") : "Vul met AI"}
        </Button>
      </div>

      {mode === "album" && albumJob && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-3 rounded-xl border border-primary/25 bg-primary/5 p-4"
        >
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <div className="text-sm font-semibold">Album AI is bezig</div>
              <div className="text-xs text-muted-foreground">
                {asText(albumJob.current_task) || asText(albumJob.status) || "Taak wordt voorbereid"}
              </div>
            </div>
            <Badge variant={albumWaiting ? "secondary" : "outline"}>
              {asText(albumJob.stage) || asText(albumJob.state) || "running"}
            </Badge>
          </div>
          <Progress value={Math.max(0, Math.min(100, asNumber(albumJob.progress, 0)))} />
          <div className="grid gap-2 text-xs text-muted-foreground sm:grid-cols-3">
            <div>
              Track{" "}
              <span className="font-medium text-foreground">
                {asText(albumJob.current_track) || "0"}/{asText(albumJob.total_tracks) || "?"}
              </span>
            </div>
            <div>
              Klaar{" "}
              <span className="font-medium text-foreground">
                {asText(albumJob.completed_tracks) || "0"}
              </span>{" "}
              · Nog{" "}
              <span className="font-medium text-foreground">
                {asText(albumJob.remaining_tracks) || "?"}
              </span>
            </div>
            <div className="flex items-center gap-1">
              <Clock3 className="size-3" />
              {albumWaiting
                ? `Wacht op ${asText(albumJob.llm_provider) || "LLM"} ${asText(albumJob.llm_wait_elapsed_s) || "0"}s`
                : asText(albumJob.current_agent) || "CrewAI"}
            </div>
          </div>
          {Array.isArray(albumJob.logs) && albumJob.logs.length > 0 && (
            <div className="rounded-md border border-border/50 bg-background/50 p-2 text-xs text-muted-foreground">
              {String(albumJob.logs[albumJob.logs.length - 1])}
            </div>
          )}
        </motion.div>
      )}

      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-3 rounded-xl border border-border/60 bg-card/40 p-4"
      >
        {warnings.length > 0 && (
          <div className="flex gap-2 text-sm">
            <AlertTriangle className="size-4 shrink-0 text-yellow-400" />
            <ul className="space-y-1">
              {warnings.map((w, i) => (
                <li key={i} className="text-yellow-200/90">{w}</li>
              ))}
            </ul>
          </div>
        )}

        <div className="space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-muted-foreground">
              <Wand2 className="size-3" /> JSON / paste block
            </div>
            <Badge variant="outline">Altijd zichtbaar</Badge>
          </div>
          <Textarea
            value={pasteEditorValue}
            onChange={(e) => updatePasteBlock(0, e.target.value)}
            rows={9}
            spellCheck={false}
            className="font-mono text-xs leading-relaxed"
          />
          <div className="flex flex-wrap gap-2">
            <Button type="button" size="sm" onClick={() => void applyPasteEditor()}>
              Pas JSON toe
            </Button>
            <Button type="button" size="sm" variant="outline" onClick={resetPasteEditor}>
              Vul met huidig formulier
            </Button>
          </div>
          {manualApplyError ? (
            <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-100">
              <p className="font-medium text-red-300">JSON fout</p>
              <p className="mt-1 whitespace-pre-wrap">{manualApplyError}</p>
            </div>
          ) : null}
          <p className="text-[11px] text-muted-foreground">
            Plak normaal één object per item. Arrays en wrappers zoals <code>songs[]</code>, <code>albums[]</code> en <code>sweeps[]</code> blijven ondersteund als bulk-import.
          </p>
        </div>

        {pasteBlocks.length > 1 && (
          <div className="space-y-1.5">
            <div className="text-xs uppercase tracking-wider text-muted-foreground">
              Bewerkbare extra paste blocks
            </div>
            {pasteBlocks.slice(1).map((b, i) => (
              <div key={i} className="space-y-2 rounded-md border border-border/40 bg-background/40 p-2 text-xs">
                <div className="font-medium">{b.label || `Block ${i + 2}`}</div>
                <Textarea
                  value={editablePasteBlocks[i + 1]?.content ?? b.content}
                  onChange={(event) => updatePasteBlock(i + 1, event.target.value)}
                  rows={7}
                  spellCheck={false}
                  className="font-mono text-[11px] leading-relaxed"
                />
                <Button type="button" size="sm" variant="outline" onClick={() => void applyPasteEditor(i + 1)}>
                  Pas dit blok toe
                </Button>
              </div>
            ))}
          </div>
        )}
      </motion.div>
    </div>
  );
}
