import * as React from "react";
import { useNavigate } from "react-router-dom";
import { useForm, Controller } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { BarChart3, ListMusic, Music4, RefreshCw } from "lucide-react";
import { z } from "zod";

import { WizardShell, FieldGroup, type WizardStepDef } from "@/components/wizard/WizardShell";
import { AIPromptStep } from "@/components/wizard/AIPromptStep";
import { MusicQueueEditor } from "@/components/wizard/MusicQueueEditor";
import { ReviewStep } from "@/components/wizard/ReviewStep";
import { GenerationAudioList } from "@/components/wizard/GenerationAudioList";
import { RenderInsightPanel } from "@/components/wizard/RenderInsightPanel";
import { AutomationFields } from "@/components/wizard/AutomationFields";
import { TagInput } from "@/components/wizard/TagInput";
import { AudioStyleSelector } from "@/components/wizard/AudioStyleSelector";
import { AudioBackendSelector } from "@/components/wizard/AudioBackendSelector";
import { AceStepAdvancedSettings } from "@/components/wizard/AceStepAdvancedSettings";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { toast } from "@/components/ui/sonner";
import {
  getLoraAdapters,
  startLoraSweepBatchJob,
  getLoraSweepJob,
  startLoraSweepJob,
  type LoraAdapter,
  type LoraSweepJob,
} from "@/lib/api";
import { customSchema, simpleDefaults, type CustomFormValues } from "@/lib/schemas";
import {
  ACE_STEP_ADVANCED_DEFAULTS,
  ACE_STEP_ADVANCED_PAYLOAD_FIELDS,
  ACE_STEP_KEY_SCALE_OPTIONS,
  ACE_STEP_TIME_SIGNATURE_OPTIONS,
  OFFICIAL_AUDIO_FORMAT_OPTIONS,
} from "@/lib/aceStepSettings";
import { ACE_STEP_LANGUAGE_OPTIONS } from "@/lib/languages";
import { isGenerationLoraAdapter, loraAdapterLabel, loraTriggerOptions } from "@/lib/lora";
import { DEFAULT_AUDIO_BACKEND, audioBackendLabel, useMlxDitForAudioBackend } from "@/lib/audioBackend";
import { extractPromptCompanion, mergePayloadWithCompanion, stripPromptCompanion, summarizeQueueEntry } from "@/lib/musicQueue";
import { mergeWizardDraft, useWizardDraft } from "@/hooks/useWizardDraft";
import { useWizardStore } from "@/store/wizard";
import { useJobsStore } from "@/store/jobs";
import { formatDuration } from "@/lib/utils";

const MODE = "lora_sweep" as const;

const SONG_MODELS = [
  ["acestep-v15-xl-sft", "ACE-Step v1.5 XL SFT (aanbevolen)"],
  ["acestep-v15-xl-base", "ACE-Step v1.5 XL Base"],
  ["acestep-v15-xl-turbo", "ACE-Step v1.5 XL Turbo"],
  ["acestep-v15-sft", "ACE-Step v1.5 SFT"],
  ["acestep-v15-base", "ACE-Step v1.5 Base"],
  ["acestep-v15-turbo", "ACE-Step v1.5 Turbo"],
  ["acestep-v15-turbo-shift1", "ACE-Step v1.5 Turbo (shift 1)"],
] as const;

const QUALITY_PROFILES = [
  ["draft", "Laag (docs-correct, volledig)"],
  ["standard", "Middel (docs standaard)"],
  ["chart_master", "Hoog (beste standaardkwaliteit)"],
] as const;

const TAG_SUGGESTIONS = [
  "rap", "hip hop", "trap", "drill", "boom bap", "g-funk", "pop", "rnb",
  "house", "techno", "cinematic", "rock", "hard drums", "808", "sub bass",
  "piano", "guitar", "strings", "analog synth", "clear English vocal",
  "dry upfront vocal", "wide stereo", "polished mix",
];

const NEG_SUGGESTIONS = [
  "mumbled vocals", "muddy mix", "weak drums", "washed out mix", "thin bass",
  "off-key vocals", "clipping", "low quality",
];

const TERMINAL_STATES = new Set(["succeeded", "success", "complete", "completed", "failed", "error", "stopped"]);
const loraSweepSchema = customSchema.extend({
  simple_description: z.string().optional().default(""),
});

function isBaseSongModel(songModel: string) {
  return songModel.endsWith("-base");
}

function normalizeLoraSweepQualityProfile(value: unknown): CustomFormValues["quality_profile"] {
  const normalized = String(value || "chart_master").trim().toLowerCase().replace(/[-\s]+/g, "_");
  if (["draft", "low", "laag", "fast", "preview", "preview_fast"].includes(normalized)) return "draft";
  if (["standard", "medium", "middle", "middel", "balanced", "balanced_pro"].includes(normalized)) return "standard";
  return "chart_master";
}

function loraSweepMlxRenderDefaults(songModel: string, qualityProfile = "chart_master") {
  const normalizedQuality = normalizeLoraSweepQualityProfile(qualityProfile);
  const turbo = songModel.includes("turbo");
  return {
    audio_backend: DEFAULT_AUDIO_BACKEND,
    quality_profile: normalizedQuality,
    inference_steps: turbo ? 8 : normalizedQuality === "chart_master" ? 64 : 50,
    guidance_scale: turbo ? 7 : 8,
    shift: 3,
    audio_format: normalizedQuality === "draft" ? "wav" : "wav32",
    infer_method: "ode" as const,
    sampler_mode: "heun" as const,
    use_adg: !turbo && normalizedQuality === "chart_master" && isBaseSongModel(songModel),
  };
}

function applyLoraSweepMlxDefaults<T extends Partial<CustomFormValues>>(values: T): T {
  const songModel = String(values.song_model || "acestep-v15-xl-sft");
  const qualityProfile = String(values.quality_profile || "chart_master");
  const renderDefaults = loraSweepMlxRenderDefaults(songModel, qualityProfile);
  return {
    ...values,
    audio_backend: DEFAULT_AUDIO_BACKEND,
    quality_profile: renderDefaults.quality_profile,
    inference_steps: renderDefaults.inference_steps,
    guidance_scale: renderDefaults.guidance_scale,
    shift: renderDefaults.shift,
    audio_format: renderDefaults.audio_format as CustomFormValues["audio_format"],
    infer_method: renderDefaults.infer_method,
    sampler_mode: renderDefaults.sampler_mode,
    use_adg: renderDefaults.use_adg,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function firstPresent(...values: unknown[]) {
  for (const value of values) {
    if (value === undefined || value === null) continue;
    if (typeof value === "string" && value.trim() === "") continue;
    return value;
  }
  return undefined;
}

function asString(value: unknown): string {
  if (Array.isArray(value)) return value.map((item) => String(item || "").trim()).filter(Boolean).join(", ");
  if (isRecord(value)) return Object.values(value).map((item) => String(item || "").trim()).filter(Boolean).join(", ");
  return value === undefined || value === null ? "" : String(value).trim();
}

function asNumberOrUndefined(value: unknown): number | undefined {
  if (value === undefined || value === null || value === "") return undefined;
  if (typeof value === "string" && value.trim().toLowerCase() === "auto") return undefined;
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function asBooleanOrUndefined(value: unknown): boolean | undefined {
  if (typeof value === "boolean") return value;
  if (typeof value !== "string") return undefined;
  const normalized = value.trim().toLowerCase();
  if (["true", "1", "yes", "on"].includes(normalized)) return true;
  if (["false", "0", "no", "off"].includes(normalized)) return false;
  return undefined;
}

function normalizeLanguageCode(value: unknown): string | undefined {
  const text = asString(value).toLowerCase();
  if (!text) return undefined;
  const map: Record<string, string> = {
    english: "en",
    dutch: "nl",
    nederlands: "nl",
    french: "fr",
    francais: "fr",
    spanish: "es",
    espanol: "es",
    portuguese: "pt",
    german: "de",
    deutsch: "de",
    arabic: "ar",
    japanese: "ja",
    korean: "ko",
    mandarin: "zh",
    chinese: "zh",
    cantonese: "yue",
    hindi: "hi",
    urdu: "ur",
    punjabi: "pa",
    italian: "it",
    polish: "pl",
    russian: "ru",
    hebrew: "he",
    instrumental: "instrumental",
  };
  if (map[text]) return map[text];
  if (/^[a-z]{2,3}(?:-[a-z]{2,4})?$/.test(text)) return text;
  return undefined;
}

function normalizeTimeSignature(value: unknown): string | undefined {
  const text = asString(value);
  if (!text || text.toLowerCase() === "auto") return undefined;
  const compact = text.replace(/\s+/g, "");
  if (compact === "4/4") return "4";
  if (compact === "3/4") return "3";
  if (compact === "2/4") return "2";
  if (compact === "6/8") return "6";
  return compact;
}

function normalizeSongModel(value: unknown): string | undefined {
  const text = asString(value);
  if (!text) return undefined;
  const normalized = text.toLowerCase().replace(/_/g, "-").replace(/\s+/g, "-");
  if (normalized.includes("xl-turbo")) return "acestep-v15-xl-turbo";
  if (normalized.includes("xl-sft")) return "acestep-v15-xl-sft";
  if (normalized.includes("xl-base")) return "acestep-v15-xl-base";
  if (normalized.includes("turbo-shift1")) return "acestep-v15-turbo-shift1";
  if (normalized.includes("turbo-shift3")) return "acestep-v15-turbo-shift3";
  if (normalized.includes("turbo")) return "acestep-v15-turbo";
  if (normalized.includes("sft")) return "acestep-v15-sft";
  if (normalized.includes("base")) return "acestep-v15-base";
  if (normalized.startsWith("acestep-")) return text;
  return undefined;
}

function assignIfPresent<T extends keyof CustomFormValues>(
  target: Partial<CustomFormValues>,
  key: T,
  value: CustomFormValues[T] | undefined,
) {
  if (value !== undefined && value !== "") target[key] = value;
}

function normalizeLoraSweepHydrationPayload(payload: Record<string, unknown>): Partial<CustomFormValues> {
  const copyPaste = isRecord(payload.copy_paste_block) ? payload.copy_paste_block : {};
  const metadata = isRecord(payload.metadata) ? payload.metadata : {};
  const copyMetadata = isRecord(copyPaste.metadata) ? copyPaste.metadata : {};
  const generation = isRecord(payload.generation_settings)
    ? payload.generation_settings
    : isRecord(payload.generation)
      ? payload.generation
      : {};
  const copyGeneration = isRecord(copyPaste.generation) ? copyPaste.generation : {};
  const advanced = isRecord(payload.advanced_generation_settings)
    ? payload.advanced_generation_settings
    : isRecord(payload.advanced_generation)
      ? payload.advanced_generation
      : {};
  const copyAdvanced = isRecord(copyPaste.advanced_generation) ? copyPaste.advanced_generation : {};
  const workflow = isRecord(copyPaste.workflow) ? copyPaste.workflow : {};

  const normalized: Partial<CustomFormValues> = {};
  const caption = asString(firstPresent(payload.ace_caption, payload.caption, copyPaste.caption, payload.tags));
  const tags = asString(firstPresent(payload.tags, payload.caption, payload.ace_caption, copyPaste.caption));
  const lyrics = asString(firstPresent(payload.lyrics, copyPaste.lyrics));
  const negativeTags = asString(firstPresent(payload.negative_tags, payload.negative_control));
  const taskType = asString(firstPresent(payload.task_type, payload.workflow_mode, workflow.workflow_mode, workflow.mode));
  const songModel = normalizeSongModel(firstPresent(payload.song_model, generation.model, copyGeneration.model));
  const duration = asNumberOrUndefined(firstPresent(payload.duration, payload.audio_duration, metadata.duration, copyMetadata.duration));
  const bpm = asNumberOrUndefined(firstPresent(payload.bpm, metadata.bpm, copyMetadata.bpm));
  const keyScale = asString(firstPresent(payload.key_scale, payload.keyscale, metadata.key_scale, metadata.keyscale, copyMetadata.key_scale, copyMetadata.keyscale));
  const timeSignature = normalizeTimeSignature(firstPresent(payload.time_signature, payload.timesignature, metadata.time_signature, metadata.timesignature, copyMetadata.time_signature, copyMetadata.timesignature));
  const vocalLanguage = normalizeLanguageCode(firstPresent(payload.vocal_language, metadata.vocal_language, copyMetadata.vocal_language, payload.target_language));

  assignIfPresent(normalized, "title", asString(firstPresent(payload.title, payload.track_name)) || undefined);
  assignIfPresent(normalized, "artist_name", asString(firstPresent(payload.artist_name, payload.artist)) || undefined);
  assignIfPresent(normalized, "caption", caption || undefined);
  assignIfPresent(normalized, "tags", tags || caption || undefined);
  assignIfPresent(normalized, "negative_tags", negativeTags || undefined);
  assignIfPresent(normalized, "lyrics", lyrics || undefined);
  assignIfPresent(normalized, "instrumental", asBooleanOrUndefined(payload.instrumental) ?? (lyrics.trim() === "[Instrumental]" ? true : undefined));
  assignIfPresent(normalized, "duration", duration);
  assignIfPresent(normalized, "bpm", bpm);
  assignIfPresent(normalized, "key_scale", keyScale && keyScale.toLowerCase() !== "auto" ? keyScale : undefined);
  assignIfPresent(normalized, "time_signature", timeSignature);
  assignIfPresent(normalized, "vocal_language", vocalLanguage);
  assignIfPresent(normalized, "song_model", songModel);
  assignIfPresent(normalized, "batch_size", asNumberOrUndefined(firstPresent(payload.batch_size, payload.variant_count, generation.batch_size, copyGeneration.batch_size)) as CustomFormValues["batch_size"] | undefined);
  assignIfPresent(normalized, "seed", asNumberOrUndefined(firstPresent(payload.seed, generation.seed, copyGeneration.seed)));
  assignIfPresent(normalized, "inference_steps", asNumberOrUndefined(firstPresent(payload.inference_steps, generation.inference_steps, copyGeneration.inference_steps)) as CustomFormValues["inference_steps"] | undefined);
  assignIfPresent(normalized, "guidance_scale", asNumberOrUndefined(firstPresent(payload.guidance_scale, generation.guidance_scale, copyGeneration.guidance_scale)));
  assignIfPresent(normalized, "shift", asNumberOrUndefined(firstPresent(payload.shift, generation.shift, copyGeneration.shift)));
  assignIfPresent(normalized, "audio_format", asString(firstPresent(payload.audio_format, generation.audio_format, copyGeneration.audio_format)) as CustomFormValues["audio_format"] | undefined);
  if (["text2music", "cover", "repaint", "extract", "lego", "complete"].includes(taskType)) {
    assignIfPresent(normalized, "task_type", taskType as CustomFormValues["task_type"]);
  }

  const advancedSource = { ...advanced, ...copyAdvanced };
  for (const key of ACE_STEP_ADVANCED_PAYLOAD_FIELDS) {
    const value = advancedSource[key] ?? payload[key];
    if (value === undefined || value === null || value === "") continue;
    // @ts-expect-error dynamic ACE-Step advanced payload hydration
    normalized[key] = value;
  }
  return normalized;
}

const LORA_SWEEP_NORMALIZED_FORM_FIELDS = new Set([
  "title",
  "artist_name",
  "caption",
  "tags",
  "negative_tags",
  "lyrics",
  "instrumental",
  "duration",
  "bpm",
  "key_scale",
  "time_signature",
  "vocal_language",
  "song_model",
  "batch_size",
  "seed",
  "inference_steps",
  "guidance_scale",
  "shift",
  "audio_format",
  "task_type",
  ...ACE_STEP_ADVANCED_PAYLOAD_FIELDS,
]);

function isExportedGenerationLora(adapter: LoraAdapter) {
  const source = String(adapter.source || "").toLowerCase();
  const path = String(adapter.path || "");
  return isGenerationLoraAdapter(adapter) && (source === "exports" || path.includes("/data/loras/"));
}

function adapterUpdatedScore(adapter: LoraAdapter) {
  const raw = adapter.updated_at || adapter.metadata?.updated_at || "";
  const parsed = Date.parse(String(raw));
  return Number.isFinite(parsed) ? parsed : 0;
}

function latestAdapterPath(adapters: LoraAdapter[]) {
  return [...adapters].sort((a, b) => {
    const byUpdated = adapterUpdatedScore(b) - adapterUpdatedScore(a);
    return byUpdated || loraAdapterLabel(a).localeCompare(loraAdapterLabel(b));
  })[0]?.path || "";
}

function sweepJobPatch(job: LoraSweepJob) {
  return {
    id: job.id,
    kind: "lora-sweep" as const,
    label: job.sweep_title || "LoRA Sweep",
    progress: job.progress || 0,
    status: job.status || job.state || "queued",
    state: job.state || "queued",
    stage: job.stage || "",
    kindLabel: "LoRA sweep",
    detailsPath: `/api/lora/sweeps/jobs/${encodeURIComponent(job.id)}`,
    logPath: `/api/lora/sweeps/jobs/${encodeURIComponent(job.id)}/log`,
    metadata: job as unknown as Record<string, unknown>,
    error: job.error || job.errors?.[0] || "",
    startedAt: job.created_at ? new Date(job.created_at).getTime() : Date.now(),
    updatedAt: job.updated_at || job.finished_at || "",
  };
}

export function LoraSweepWizard() {
  const navigate = useNavigate();
  const addJob = useJobsStore((s) => s.addJob);
  const openJob = useJobsStore((s) => s.openJob);
  const setResult = useWizardStore((s) => s.setResult);
  const setPasteBlocks = useWizardStore((s) => s.setPasteBlocks);
  const lastResult = useWizardStore((s) => s.lastResult[MODE]);
  const warnings = useWizardStore((s) => s.warnings[MODE]) ?? [];
  const storedQueue = useWizardStore((s) => s.queues[MODE]) ?? [];
  const storedCompanion = useWizardStore((s) => s.companions[MODE]) ?? {};
  const setHydration = useWizardStore((s) => s.setHydration);
  const draft = useWizardStore((s) => s.drafts[MODE]);
  const defaults = React.useMemo<CustomFormValues>(
    () => ({
      ...simpleDefaults,
      ...(ACE_STEP_ADVANCED_DEFAULTS as Partial<CustomFormValues>),
      task_type: "text2music",
      title: "LoRA Sweep",
      audio_backend: DEFAULT_AUDIO_BACKEND,
      quality_profile: "chart_master",
      inference_steps: 64,
      guidance_scale: 8,
      shift: 3,
      audio_format: "wav32",
      infer_method: "ode",
      sampler_mode: "heun",
      use_adg: false,
      batch_size: 1,
      use_lora: false,
      lora_adapter_path: "",
      lora_adapter_name: "",
    }),
    [],
  );
  const initialValues = React.useMemo(
    () => applyLoraSweepMlxDefaults(mergeWizardDraft<CustomFormValues>(defaults, draft)),
    [defaults, draft],
  );

  const form = useForm<CustomFormValues>({
    resolver: zodResolver(loraSweepSchema),
    defaultValues: initialValues,
    mode: "onChange",
  });

  const [step, setStep] = React.useState(0);
  const [isStarting, setIsStarting] = React.useState(false);
  const [queue, setQueue] = React.useState<Record<string, unknown>[]>(storedQueue);
  const [editingQueueIndex, setEditingQueueIndex] = React.useState(-1);
  const [companion, setCompanion] = React.useState<Record<string, unknown>>(storedCompanion);
  const [includeBaseline, setIncludeBaseline] = React.useState(false);
  const [loraScale, setLoraScale] = React.useState(1);
  const [selectedAdapterPaths, setSelectedAdapterPaths] = React.useState<string[]>([]);
  const [adapterSelectionTouched, setAdapterSelectionTouched] = React.useState(false);
  const [activeJob, setActiveJob] = React.useState<LoraSweepJob | null>(null);
  const values = form.watch();
  const draftState = useWizardDraft(MODE, form);

  const adaptersQuery = useQuery({
    queryKey: ["lora", "sweep-adapters"],
    queryFn: getLoraAdapters,
    refetchInterval: 10_000,
  });
  const adapters = React.useMemo(
    () => (adaptersQuery.data?.adapters || []).filter(isExportedGenerationLora),
    [adaptersQuery.data?.adapters],
  );
  const selectedAdapterPathSet = React.useMemo(() => new Set(selectedAdapterPaths), [selectedAdapterPaths]);
  const selectedAdapters = React.useMemo(
    () => adapters.filter((adapter) => selectedAdapterPathSet.has(adapter.path)),
    [adapters, selectedAdapterPathSet],
  );
  const selectedAdapterNames = React.useMemo(() => selectedAdapters.map(loraAdapterLabel), [selectedAdapters]);
  const totalRenders = selectedAdapters.length * values.batch_size + (includeBaseline ? values.batch_size : 0);
  const formValidation = React.useMemo(() => loraSweepSchema.safeParse(values), [values]);
  const formIssueMessages = React.useMemo(() => {
    if (formValidation.success) return [];
    return formValidation.error.issues.slice(0, 6).map((issue) => {
      const path = issue.path.join(".") || "formulier";
      return `${path}: ${issue.message}`;
    });
  }, [formValidation]);
  const reviewBlockers = React.useMemo(() => {
    const blockers = [...formIssueMessages];
    if (!selectedAdapters.length && !includeBaseline) blockers.push("Kies minstens één LoRA of zet de baseline aan.");
    return blockers;
  }, [formIssueMessages, includeBaseline, selectedAdapters.length]);
  const canStartSweep = reviewBlockers.length === 0;

  React.useEffect(() => {
    setSelectedAdapterPaths((current) => {
      const available = new Set(adapters.map((adapter) => adapter.path));
      const kept = current.filter((path) => available.has(path));
      if (kept.length || adapterSelectionTouched || !adapters.length) return kept;
      const latest = latestAdapterPath(adapters);
      return latest ? [latest] : [];
    });
  }, [adapters, adapterSelectionTouched]);

  const toggleAdapterPath = React.useCallback((path: string, checked: boolean) => {
    setAdapterSelectionTouched(true);
    setSelectedAdapterPaths((current) => {
      if (checked) return Array.from(new Set([...current, path]));
      return current.filter((item) => item !== path);
    });
  }, []);

  const selectAllAdapters = React.useCallback(() => {
    setAdapterSelectionTouched(true);
    setSelectedAdapterPaths(adapters.map((adapter) => adapter.path));
  }, [adapters]);

  const selectLatestAdapter = React.useCallback(() => {
    setAdapterSelectionTouched(true);
    const latest = latestAdapterPath(adapters);
    setSelectedAdapterPaths(latest ? [latest] : []);
  }, [adapters]);

  const clearSelectedAdapters = React.useCallback(() => {
    setAdapterSelectionTouched(true);
    setSelectedAdapterPaths([]);
  }, []);

  React.useEffect(() => {
    if (!activeJob?.id || TERMINAL_STATES.has(String(activeJob.state || "").toLowerCase())) return;
    const timer = window.setInterval(async () => {
      try {
        const resp = await getLoraSweepJob(activeJob.id);
        if (resp.job) {
          setActiveJob(resp.job);
          addJob(sweepJobPatch(resp.job));
          if (TERMINAL_STATES.has(String(resp.job.state || "").toLowerCase())) {
            setResult(MODE, resp.job as unknown as Record<string, unknown>);
          }
        }
      } catch {
        // Job tracker will surface detail fetch errors; keep the wizard calm.
      }
    }, 2500);
    return () => window.clearInterval(timer);
  }, [activeJob?.id, activeJob?.state, addJob, setResult]);

  const setAdvancedValue = (key: string, value: unknown) => {
    form.setValue(key as keyof CustomFormValues, value as never, {
      shouldDirty: true,
      shouldValidate: true,
    });
  };

  const hydrate = (payload: Record<string, unknown>) => {
    setCompanion(extractPromptCompanion(payload));
    const next: Partial<CustomFormValues> = normalizeLoraSweepHydrationPayload(payload);
    for (const [k, v] of Object.entries(payload)) {
      if (k in form.getValues() || k === "simple_description") {
        if (LORA_SWEEP_NORMALIZED_FORM_FIELDS.has(k)) continue;
        // @ts-expect-error dynamic AI payload hydration
        next[k] = next[k as keyof CustomFormValues] ?? v;
      }
    }
    next.use_lora = false;
    next.lora_adapter_path = "";
    next.lora_adapter_name = "";
    const merged = applyLoraSweepMlxDefaults({ ...form.getValues(), ...next });
    form.reset(merged);
    draftState.saveNow(merged);
    return merged;
  };

  const buildCompanion = React.useCallback(() => ({ ...companion }), [companion]);

  const buildPayload = () => {
    const v = form.getValues();
    const advanced: Record<string, unknown> = {};
    for (const key of ACE_STEP_ADVANCED_PAYLOAD_FIELDS) {
      const value = v[key as keyof CustomFormValues];
      if (value === undefined || value === "") continue;
      advanced[key] = value;
    }
    return {
      task_type: v.task_type,
      title: v.title,
      artist_name: v.artist_name,
      caption: v.caption,
      style_profile: v.style_profile,
      tags: v.tags,
      negative_tags: v.negative_tags,
      lyrics: v.instrumental ? "[Instrumental]" : v.lyrics,
      instrumental: v.instrumental,
      audio_duration: v.duration,
      duration: v.duration,
      bpm: v.bpm,
      key_scale: v.key_scale,
      time_signature: v.time_signature,
      vocal_language: v.vocal_language,
      song_model: v.song_model,
      audio_backend: DEFAULT_AUDIO_BACKEND,
      use_mlx_dit: useMlxDitForAudioBackend(DEFAULT_AUDIO_BACKEND),
      quality_profile: v.quality_profile,
      seed: v.seed,
      inference_steps: v.inference_steps,
      guidance_scale: v.guidance_scale,
      shift: v.shift,
      audio_format: v.audio_format,
      batch_size: v.batch_size,
      variant_count: v.batch_size,
      use_lora: false,
      lora_adapter_path: "",
      lora_adapter_name: "",
      use_lora_trigger: false,
      lora_trigger_tag: "",
      lora_scale: 0,
      auto_song_art: v.auto_song_art,
      auto_album_art: false,
      auto_video_clip: v.auto_video_clip,
      art_prompt: v.art_prompt,
      video_prompt: v.video_prompt,
      ...advanced,
    };
  };

  const buildSweepRequest = React.useCallback(() => ({
    sweep_title: form.getValues("title") || "LoRA Sweep",
    adapter_paths: selectedAdapters.map((adapter) => adapter.path),
    selected_adapters: selectedAdapters.map((adapter) => ({
      path: adapter.path,
      name: loraAdapterLabel(adapter),
    })),
    render_payload: buildPayload(),
    variant_count: form.getValues("batch_size"),
    include_baseline: includeBaseline,
    lora_scale: loraScale,
    trigger_mode: "auto",
    stop_on_error: false,
  }), [buildPayload, form, includeBaseline, loraScale, selectedAdapters]);

  const buildDraftPayload = React.useCallback(
    () => mergePayloadWithCompanion(buildSweepRequest(), buildCompanion()),
    [buildCompanion, buildSweepRequest],
  );

  const syncQueueState = React.useCallback(
    (nextQueue: Record<string, unknown>[]) => {
      setQueue(nextQueue);
      setHydration(MODE, {
        payload: buildSweepRequest(),
        warnings,
        companion: buildCompanion(),
        queue: nextQueue,
      });
    },
    [buildCompanion, buildSweepRequest, setHydration, warnings],
  );

  const resetForNextDraft = React.useCallback(() => {
    const current = form.getValues();
    const next: CustomFormValues = {
      ...current,
      title: "",
      artist_name: "",
      caption: "",
      style_profile: "auto",
      tags: "",
      negative_tags: "",
      lyrics: undefined,
      art_prompt: "",
      video_prompt: "",
      payload_gate_status: "",
      payload_gate_passed: false,
      payload_gate_blocking_issues: [],
      payload_quality_gate: {},
      rap_quality_report: {},
      rap_rewrite_status: "",
      rap_blocking_issues: [],
      rap_strengths: [],
      rap_revision_focus: [],
    };
    form.reset(next);
    setCompanion({});
    setEditingQueueIndex(-1);
    draftState.saveNow(next);
    setPasteBlocks(MODE, [{ label: "Huidige wizard JSON", content: "" }]);
  }, [draftState, form, setPasteBlocks]);

  const addCurrentToQueue = React.useCallback(() => {
    const entry = buildDraftPayload();
    const nextQueue = [...queue];
    if (editingQueueIndex >= 0 && editingQueueIndex < nextQueue.length) {
      nextQueue[editingQueueIndex] = entry;
    } else {
      nextQueue.push(entry);
    }
    syncQueueState(nextQueue);
    resetForNextDraft();
  }, [buildDraftPayload, editingQueueIndex, queue, resetForNextDraft, syncQueueState]);

  const queueItemsForSubmit = React.useMemo(() => {
    const current = buildDraftPayload();
    if (editingQueueIndex >= 0) {
      return queue.map((item, index) => (index === editingQueueIndex ? current : item));
    }
    const hasMeaningfulDraft =
      Boolean((values.title ?? "").trim()) ||
      Boolean((values.caption ?? "").trim()) ||
      Boolean((values.tags ?? "").trim()) ||
      Boolean((values.lyrics ?? "").trim()) ||
      selectedAdapters.length > 0;
    return hasMeaningfulDraft ? [...queue, current] : queue;
  }, [buildDraftPayload, editingQueueIndex, queue, selectedAdapters.length, values.caption, values.lyrics, values.tags, values.title]);

  const handleFinish = async () => {
    setIsStarting(true);
    try {
      const queued = queueItemsForSubmit;
      if (queued.length > 1) {
        const resp = await startLoraSweepBatchJob({
          batch_title: values.title || "Queued LoRA sweeps",
          stop_on_error: false,
          sweeps: queued.map((item) => stripPromptCompanion(item)),
        });
        if (!resp.success || !resp.job_id) throw new Error(resp.error || "LoRA sweep queue starten mislukt");
        addJob({
          id: resp.job_id,
          kind: "lora-sweep-batch",
          label: values.title || "LoRA sweep batch",
          progress: resp.job?.progress || 0,
          status: resp.job?.status || resp.job?.state || "queued",
          state: resp.job?.state || "queued",
          stage: resp.job?.stage || "",
          kindLabel: "LoRA sweep batch",
          detailsPath: `/api/lora/sweep-batches/jobs/${encodeURIComponent(resp.job_id)}`,
          logPath: `/api/lora/sweep-batches/jobs/${encodeURIComponent(resp.job_id)}/log`,
          metadata: (resp.job as Record<string, unknown> | undefined) ?? {},
          error: resp.error || "",
          startedAt: Date.now(),
          deletePath: `/api/lora/sweep-batches/jobs/${encodeURIComponent(resp.job_id)}`,
        });
        openJob(resp.job_id);
        toast.success("LoRA sweep queue gestart.");
        return;
      }
      if (!selectedAdapters.length && !includeBaseline) {
        toast.error("Kies minstens één LoRA of zet de baseline aan.");
        return;
      }
      const resp = await startLoraSweepJob(buildSweepRequest());
      if (!resp.success || !resp.job_id) throw new Error(resp.error || "LoRA Sweep starten mislukt");
      const job = resp.job || ({ id: resp.job_id, sweep_title: values.title || "LoRA Sweep", state: "queued" } as LoraSweepJob);
      setActiveJob(job);
      setResult(MODE, job as unknown as Record<string, unknown>);
      addJob(sweepJobPatch(job));
      setStep(6);
      toast.success("LoRA Sweep gestart.");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "LoRA Sweep starten mislukt");
    } finally {
      setIsStarting(false);
    }
  };

  const steps: WizardStepDef[] = [
    {
      key: "ai",
      title: "AI song fill",
      description: "Laat AI één complete song-payload maken; de sweep rendert daarna de gekozen LoRAs met exact die song.",
      isValid: true,
      render: () => (
        <AIPromptStep
          mode="custom"
          placeholder="Maak een volledige rap/pop/club song met caption, tags, lyrics en metadata voor een LoRA sweep..."
          currentPayload={buildPayload()}
          onHydrated={hydrate}
          onManualApply={() => {
            setStep(1);
          }}
        />
      ),
    },
    {
      key: "song",
      title: "Song",
      isValid: true,
      render: () => (
        <div className="space-y-4">
          <FieldGroup title="Naam">
            <div className="grid gap-3 sm:grid-cols-3">
              <div className="space-y-1.5">
                <Label>Titel</Label>
                <Input {...form.register("title")} />
              </div>
              <div className="space-y-1.5">
                <Label>Artiest</Label>
                <Input {...form.register("artist_name")} />
              </div>
              <div className="space-y-1.5">
                <Label>Taal</Label>
                <Controller
                  control={form.control}
                  name="vocal_language"
                  render={({ field }) => (
                    <Select value={field.value} onValueChange={field.onChange}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {ACE_STEP_LANGUAGE_OPTIONS.map(([code, name]) => (
                          <SelectItem key={code} value={code}>{name}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>
            </div>
          </FieldGroup>
          <FieldGroup title="Vocals">
            <div className="flex items-center justify-between rounded-md border bg-background/35 p-3">
              <div>
                <Label>Instrumentaal</Label>
                <p className="text-xs text-muted-foreground">Aan = lyrics worden vervangen door [Instrumental].</p>
              </div>
              <Controller
                control={form.control}
                name="instrumental"
                render={({ field }) => <Switch checked={field.value} onCheckedChange={field.onChange} />}
              />
            </div>
          </FieldGroup>
        </div>
      ),
    },
    {
      key: "sound",
      title: "Sound & lyrics",
      isValid: true,
      render: () => (
        <div className="space-y-4">
          <FieldGroup title="Caption">
            <AudioStyleSelector
              value={values.style_profile}
              onChange={(value) => form.setValue("style_profile", value, { shouldValidate: true })}
            />
            <Textarea rows={2} {...form.register("caption")} />
          </FieldGroup>
          <FieldGroup title="Tags">
            <Controller
              control={form.control}
              name="tags"
              render={({ field }) => (
                <TagInput value={field.value ?? ""} onChange={field.onChange} suggestions={TAG_SUGGESTIONS} />
              )}
            />
          </FieldGroup>
          <FieldGroup title="Negative tags">
            <Controller
              control={form.control}
              name="negative_tags"
              render={({ field }) => (
                <TagInput value={field.value ?? ""} onChange={field.onChange} suggestions={NEG_SUGGESTIONS} variant="negative" />
              )}
            />
          </FieldGroup>
          {!values.instrumental && (
            <FieldGroup title="Lyrics">
              <Textarea rows={16} className="font-mono text-xs leading-relaxed" {...form.register("lyrics")} />
            </FieldGroup>
          )}
        </div>
      ),
    },
    {
      key: "render",
      title: "Render & sweep",
      isValid: true,
      render: () => (
        <div className="space-y-4">
          <FieldGroup title="LoRA Sweep" description="Render dezelfde song met de gekozen audio-LoRAs uit app/data/loras.">
            <div className="grid gap-3 sm:grid-cols-4">
              <div className="rounded-md border bg-background/35 p-3">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Gekozen LoRAs</p>
                <p className="mt-1 font-mono text-lg">{selectedAdapters.length} / {adapters.length}</p>
              </div>
              <div className="rounded-md border bg-background/35 p-3">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Variaties</p>
                <p className="mt-1 font-mono text-lg">{values.batch_size}</p>
              </div>
              <div className="rounded-md border bg-background/35 p-3">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Baseline</p>
                <p className="mt-1 font-mono text-lg">{includeBaseline ? "aan" : "uit"}</p>
              </div>
              <div className="rounded-md border bg-background/35 p-3">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Totaal</p>
                <p className="mt-1 font-mono text-lg">{totalRenders}</p>
              </div>
            </div>
            <div className="mt-3 flex items-center justify-between rounded-md border bg-background/35 p-3">
              <div>
                <Label>No-LoRA baseline</Label>
                <p className="text-xs text-muted-foreground">Optioneel, standaard uit.</p>
              </div>
              <Switch checked={includeBaseline} onCheckedChange={setIncludeBaseline} />
            </div>
          </FieldGroup>

          <FieldGroup title="Model & kwaliteit">
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1.5">
                <Label>Fallback song model</Label>
                <Controller
                  control={form.control}
                  name="song_model"
                  render={({ field }) => (
                    <Select
                      value={field.value}
                      onValueChange={(nextModel) => {
                        field.onChange(nextModel);
                        const next = loraSweepMlxRenderDefaults(nextModel, values.quality_profile);
                        form.setValue("inference_steps", next.inference_steps);
                        form.setValue("guidance_scale", next.guidance_scale);
                        form.setValue("shift", next.shift);
                        form.setValue("audio_format", next.audio_format as CustomFormValues["audio_format"]);
                        form.setValue("infer_method", next.infer_method);
                        form.setValue("sampler_mode", next.sampler_mode);
                        form.setValue("use_adg", next.use_adg);
                      }}
                    >
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {SONG_MODELS.map(([id, label]) => <SelectItem key={id} value={id}>{label}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>
              <div className="space-y-1.5">
                <Label>Kwaliteit</Label>
                <Controller
                  control={form.control}
                  name="quality_profile"
                  render={({ field }) => (
                    <Select
                      value={field.value}
                      onValueChange={(nextQuality) => {
                        field.onChange(nextQuality);
                        const next = loraSweepMlxRenderDefaults(values.song_model, nextQuality);
                        form.setValue("inference_steps", next.inference_steps);
                        form.setValue("guidance_scale", next.guidance_scale);
                        form.setValue("shift", next.shift);
                        form.setValue("audio_format", next.audio_format as CustomFormValues["audio_format"]);
                        form.setValue("infer_method", next.infer_method);
                        form.setValue("sampler_mode", next.sampler_mode);
                        form.setValue("use_adg", next.use_adg);
                      }}
                    >
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {QUALITY_PROFILES.map(([id, label]) => <SelectItem key={id} value={id}>{label}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>
              <AudioBackendSelector
                value={DEFAULT_AUDIO_BACKEND}
                onChange={() => form.setValue("audio_backend", DEFAULT_AUDIO_BACKEND, { shouldValidate: true })}
              />
            </div>
          </FieldGroup>

          <AutomationFields control={form.control} register={form.register} values={values} />

          <FieldGroup title="Variaties en inference">
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-3">
                <div className="flex items-baseline justify-between">
                  <Label>Variaties per LoRA</Label>
                  <span className="font-mono text-xs">{values.batch_size}</span>
                </div>
                <Controller
                  control={form.control}
                  name="batch_size"
                  render={({ field }) => (
                    <Slider value={[field.value]} min={1} max={8} step={1} onValueChange={(v) => field.onChange(v[0] ?? 1)} />
                  )}
                />
              </div>
              <div className="space-y-3">
                <div className="flex items-baseline justify-between">
                  <Label>LoRA scale</Label>
                  <span className="font-mono text-xs">{loraScale.toFixed(2)}</span>
                </div>
                <Slider value={[loraScale]} min={0} max={1} step={0.05} onValueChange={(v) => setLoraScale(v[0] ?? 1)} />
              </div>
              <div className="space-y-3">
                <div className="flex items-baseline justify-between">
                  <Label>Inference steps</Label>
                  <span className="font-mono text-xs">{values.inference_steps}</span>
                </div>
                <Controller control={form.control} name="inference_steps" render={({ field }) => <Slider value={[field.value]} min={4} max={100} step={1} onValueChange={(v) => field.onChange(v[0])} />} />
              </div>
              <div className="space-y-3">
                <div className="flex items-baseline justify-between">
                  <Label>Guidance scale</Label>
                  <span className="font-mono text-xs">{values.guidance_scale.toFixed(1)}</span>
                </div>
                <Controller control={form.control} name="guidance_scale" render={({ field }) => <Slider value={[field.value]} min={1} max={15} step={0.1} onValueChange={(v) => field.onChange(v[0])} />} />
              </div>
              <div className="space-y-1.5">
                <Label>Audio format</Label>
                <Controller
                  control={form.control}
                  name="audio_format"
                  render={({ field }) => (
                    <Select value={field.value} onValueChange={field.onChange}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {OFFICIAL_AUDIO_FORMAT_OPTIONS.map(([value, label]) => <SelectItem key={value} value={value}>{label}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>
              <div className="space-y-1.5">
                <Label>Seed</Label>
                <Input type="number" placeholder="-1 voor deterministic sweep seeds" {...form.register("seed", { valueAsNumber: true })} />
              </div>
            </div>
          </FieldGroup>

          <FieldGroup title="BPM, key & maatsoort">
            <div className="grid gap-3 sm:grid-cols-3">
              <Input type="number" min={40} max={300} placeholder="BPM auto" {...form.register("bpm", { valueAsNumber: true })} />
              <Controller
                control={form.control}
                name="key_scale"
                render={({ field }) => (
                  <Select value={field.value || "auto"} onValueChange={(value) => field.onChange(value === "auto" ? undefined : value)}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {ACE_STEP_KEY_SCALE_OPTIONS.map((value) => <SelectItem key={value} value={value}>{value === "auto" ? "Auto" : value}</SelectItem>)}
                    </SelectContent>
                  </Select>
                )}
              />
              <Controller
                control={form.control}
                name="time_signature"
                render={({ field }) => (
                  <Select value={field.value || "auto"} onValueChange={(value) => field.onChange(value === "auto" ? undefined : value)}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {ACE_STEP_TIME_SIGNATURE_OPTIONS.map(([value, label]) => <SelectItem key={value || "auto"} value={value || "auto"}>{label}</SelectItem>)}
                    </SelectContent>
                  </Select>
                )}
              />
            </div>
          </FieldGroup>

          <FieldGroup title="Duur">
            <div className="flex items-baseline justify-between">
              <Label>Duur</Label>
              <span className="font-mono text-sm">{formatDuration(values.duration)}</span>
            </div>
            <Controller control={form.control} name="duration" render={({ field }) => <Slider value={[field.value ?? 60]} min={20} max={600} step={5} onValueChange={(v) => field.onChange(v[0] ?? 60)} />} />
          </FieldGroup>

          <FieldGroup title="Official ACE-Step controls">
            <AceStepAdvancedSettings values={values} onChange={setAdvancedValue} />
          </FieldGroup>
        </div>
      ),
    },
    {
      key: "adapters",
      title: "Adapters",
      isValid: selectedAdapters.length > 0 || includeBaseline,
      render: () => (
        <FieldGroup title="Selecteer audio-LoRAs" description="Alleen adapters die fysiek in app/data/loras staan en generation-loadable zijn.">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-sm text-muted-foreground">
              {selectedAdapters.length} van {adapters.length} LoRA{adapters.length === 1 ? "" : "s"} geselecteerd.
            </p>
            <div className="flex flex-wrap items-center gap-2">
              <Button type="button" variant="outline" size="sm" onClick={selectLatestAdapter} disabled={!adapters.length}>
                Nieuwste
              </Button>
              <Button type="button" variant="outline" size="sm" onClick={selectAllAdapters} disabled={!adapters.length}>
                Alles
              </Button>
              <Button type="button" variant="ghost" size="sm" onClick={clearSelectedAdapters} disabled={!selectedAdapters.length}>
                Geen
              </Button>
              <Button type="button" variant="outline" size="sm" onClick={() => void adaptersQuery.refetch()} className="gap-2">
                <RefreshCw className="size-4" />
                Ververs
              </Button>
            </div>
          </div>
          <div className="mt-3 max-h-96 space-y-2 overflow-auto pr-1">
            {adapters.map((adapter) => {
              const triggers = loraTriggerOptions(adapter);
              const selected = selectedAdapterPathSet.has(adapter.path);
              return (
                <div key={adapter.path} className={`rounded-md border p-3 ${selected ? "border-primary/60 bg-primary/5" : "bg-background/35"}`}>
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium">{loraAdapterLabel(adapter)}</p>
                      <p className="truncate text-xs text-muted-foreground">{adapter.path}</p>
                    </div>
                    <div className="flex flex-wrap items-center gap-1">
                      <Badge variant="outline">{adapter.song_model || adapter.model_variant || "auto model"}</Badge>
                      <Badge variant="secondary">{triggers[0] || "no trigger"}</Badge>
                      <Switch
                        checked={selected}
                        onCheckedChange={(checked) => toggleAdapterPath(adapter.path, checked)}
                        aria-label={`Selecteer ${loraAdapterLabel(adapter)}`}
                      />
                    </div>
                  </div>
                </div>
              );
            })}
            {!adapters.length && (
              <p className="rounded-md border border-amber-500/30 bg-amber-500/10 p-3 text-sm text-amber-900 dark:text-amber-100">
                Geen generation-loadable audio-LoRAs gevonden in app/data/loras.
              </p>
            )}
          </div>
        </FieldGroup>
      ),
    },
    {
      key: "review",
      title: "Review & start",
      isValid: canStartSweep,
      render: () => (
        <div className="space-y-4">
          <ReviewStep
            payload={{
              sweep_title: values.title,
              variant_count: values.batch_size,
              include_baseline: includeBaseline,
              lora_scale: loraScale,
              selected_lora_count: selectedAdapters.length,
              selected_loras: selectedAdapterNames,
              total_renders: totalRenders,
              render_payload: buildPayload(),
            }}
            warnings={warnings}
            blockingIssues={reviewBlockers}
            queueSummary={{
              queuedItems: queue.length,
              totalRenders: queueItemsForSubmit.reduce((sum, item) => {
                const adapterCount = Array.isArray(item.selected_adapters) ? item.selected_adapters.length : Number(item.selected_lora_count || 0);
                const variantCount = Number(item.variant_count || 1);
                const baselineCount = item.include_baseline ? variantCount : 0;
                return sum + adapterCount * variantCount + baselineCount;
              }, 0),
            }}
            companion={buildCompanion()}
            primaryFields={[
              { key: "sweep_title", label: "Sweep" },
              { key: "variant_count", label: "Variaties per LoRA" },
              { key: "include_baseline", label: "Baseline" },
              { key: "lora_scale", label: "LoRA scale" },
              { key: "selected_lora_count", label: "Gekozen LoRAs" },
              { key: "total_renders", label: "Totaal renders" },
              { key: "render_payload.song_model", label: "Fallback model" },
              { key: "render_payload.audio_backend", label: "Backend", format: audioBackendLabel },
            ]}
          />
          <MusicQueueEditor
            items={queueItemsForSubmit.map((item, index) => summarizeQueueEntry(item, `Sweep ${index + 1}`))}
            activeIndex={editingQueueIndex}
            onSelect={(index) => {
              const selected = queueItemsForSubmit[index];
              if (!selected) return;
              setEditingQueueIndex(index < queue.length ? index : -1);
              setCompanion(extractPromptCompanion(selected));
              const renderPayload =
                selected.render_payload && typeof selected.render_payload === "object" && !Array.isArray(selected.render_payload)
                  ? (selected.render_payload as Record<string, unknown>)
                  : selected;
              hydrate(renderPayload);
              setPasteBlocks(MODE, [
                { label: "Huidige wizard JSON", content: JSON.stringify(selected, null, 2) },
              ]);
              setIncludeBaseline(Boolean(selected.include_baseline));
              setLoraScale(Number(selected.lora_scale || 1));
              const adapterPaths = Array.isArray(selected.adapter_paths)
                ? selected.adapter_paths.map((value) => String(value))
                : [];
              if (adapterPaths.length > 0) setSelectedAdapterPaths(adapterPaths);
            }}
            onRemove={(index) => {
              const nextQueue = queue.filter((_, itemIndex) => itemIndex !== index);
              syncQueueState(nextQueue);
              if (editingQueueIndex === index) setEditingQueueIndex(-1);
            }}
            onDuplicate={(index) => {
              const selected = queueItemsForSubmit[index];
              if (!selected) return;
              const nextQueue = [...queue, selected];
              syncQueueState(nextQueue);
            }}
            onMove={(from, direction) => {
              const nextQueue = [...queue];
              const to = from + direction;
              if (from < 0 || from >= nextQueue.length || to < 0 || to >= nextQueue.length) return;
              const [item] = nextQueue.splice(from, 1);
              nextQueue.splice(to, 0, item);
              syncQueueState(nextQueue);
            }}
          />
          {reviewBlockers.length > 0 && (
            <FieldGroup title="Start blokkades">
              <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
                {reviewBlockers.map((blocker) => (
                  <li key={blocker}>{blocker}</li>
                ))}
              </ul>
            </FieldGroup>
          )}
          <RenderInsightPanel payload={buildPayload()} warnings={warnings} />
        </div>
      ),
    },
    {
      key: "result",
      title: "Sweep resultaat",
      isValid: true,
      hidden: !activeJob && !lastResult,
      render: () => {
        const job = activeJob || (lastResult as unknown as LoraSweepJob | undefined);
        const rows = job?.items?.length ? job.items : job?.results || [];
        return (
          <div className="space-y-4">
            {job && (
              <FieldGroup title={job.sweep_title || "LoRA Sweep"}>
                <div className="grid gap-3 sm:grid-cols-4">
                  <div className="rounded-md border bg-background/35 p-3">
                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Status</p>
                    <p className="mt-1 text-sm">{job.status || job.state}</p>
                  </div>
                  <div className="rounded-md border bg-background/35 p-3">
                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Progress</p>
                    <p className="mt-1 font-mono text-sm">{job.progress || 0}%</p>
                  </div>
                  <div className="rounded-md border bg-background/35 p-3">
                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Audios</p>
                    <p className="mt-1 font-mono text-sm">{job.generated_audio_count || 0} / {job.expected_audio_count || 0}</p>
                  </div>
                  <div className="rounded-md border bg-background/35 p-3">
                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Groups</p>
                    <p className="mt-1 font-mono text-sm">{job.completed_items || 0} / {job.total_items || 0}</p>
                  </div>
                </div>
                {job.id && (
                  <Button type="button" variant="outline" className="mt-3 gap-2" onClick={() => openJob(job.id)}>
                    <ListMusic className="size-4" />
                    Open jobdetails
                  </Button>
                )}
              </FieldGroup>
            )}
            <div className="space-y-3">
              {rows.map((row, index) => {
                const variants = Array.isArray(row.variants) ? row.variants : [];
                const readyVariants = variants.filter((variant) => String(variant.state || "").toLowerCase() === "succeeded").length;
                const totalVariants = row.variant_count || variants.length || values.batch_size;
                return (
                  <div key={`${row.item_id || index}-${row.adapter_path || row.adapter_name}`} className="rounded-md border bg-background/35 p-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="min-w-0">
                        <p className="truncate text-sm font-medium">{row.adapter_name || `LoRA ${index + 1}`}</p>
                        <p className="truncate text-xs text-muted-foreground">
                          {row.song_model || "auto model"} · {readyVariants}/{totalVariants} varianten klaar
                        </p>
                      </div>
                      <Badge variant={String(row.state).toLowerCase() === "failed" ? "destructive" : String(row.state).toLowerCase() === "succeeded" ? "default" : "outline"}>
                        {row.status || row.state || "queued"}
                      </Badge>
                    </div>
                    {variants.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {variants.map((variant) => {
                          const state = String(variant.state || "").toLowerCase();
                          return (
                            <Badge
                              key={`${row.item_id || index}-variant-${variant.variant_index || variant.seed}`}
                              variant={state === "failed" ? "destructive" : state === "succeeded" ? "default" : state === "running" ? "secondary" : "outline"}
                              className="font-mono text-[10px]"
                            >
                              v{variant.variant_index || "?"}: {variant.variant_seed || variant.seed || "seed"} · {variant.status || variant.state || "queued"}
                            </Badge>
                          );
                        })}
                      </div>
                    )}
                    {row.error && <p className="mt-2 rounded-md bg-destructive/10 p-2 text-xs text-destructive">{row.error}</p>}
                    {row.result && (
                      <GenerationAudioList
                        result={row.result as Record<string, unknown>}
                        title={row.adapter_name || `LoRA ${index + 1}`}
                        artist="LoRA Sweep"
                        className="mt-3 space-y-2"
                      />
                    )}
                  </div>
                );
              })}
            </div>
            <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="flex flex-wrap items-center gap-2">
              <Button variant="outline" onClick={() => navigate("/library")} className="gap-2">
                <Music4 className="size-4" />
                Open library
              </Button>
              <Button variant="ghost" onClick={() => { form.reset(defaults); draftState.clear(); setActiveJob(null); setStep(0); }}>
                Nieuwe sweep
              </Button>
            </motion.div>
          </div>
        );
      },
    },
  ];

  return (
    <WizardShell
      title="LoRA Sweep"
      subtitle="Custom-style song bouwen, daarna renderen met de geselecteerde audio-LoRAs in de LoRA-map."
      steps={steps}
      step={step}
      onStepChange={setStep}
      onFinish={handleFinish}
      isFinishing={isStarting || Boolean(activeJob && !TERMINAL_STATES.has(String(activeJob.state || "").toLowerCase()))}
      finishLabel={isStarting ? "Start…" : queueItemsForSubmit.length > 1 ? "Start sweep queue" : "Start LoRA Sweep"}
      secondaryFinishAction={{
        label: editingQueueIndex >= 0 ? "Werk queue-item bij" : "Nog een toevoegen",
        onClick: addCurrentToQueue,
        disabled: !canStartSweep || isStarting,
      }}
    />
  );
}
