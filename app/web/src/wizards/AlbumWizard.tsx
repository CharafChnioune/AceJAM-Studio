import * as React from "react";
import { useNavigate } from "react-router-dom";
import { useForm, Controller } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import {
  Disc3, Music4, Trash2, Plus, Loader2, Sparkles, Video,
} from "lucide-react";

import { WizardShell, FieldGroup, type WizardStepDef } from "@/components/wizard/WizardShell";
import { AIPromptStep } from "@/components/wizard/AIPromptStep";
import { LoraSelector } from "@/components/wizard/LoraSelector";
import { AutomationFields } from "@/components/wizard/AutomationFields";
import { ReviewStep } from "@/components/wizard/ReviewStep";
import { AudioStyleSelector } from "@/components/wizard/AudioStyleSelector";
import { AudioBackendSelector } from "@/components/wizard/AudioBackendSelector";
import { WaveformPlayer } from "@/components/audio/WaveformPlayer";
import { MfluxArtMaker } from "@/components/mflux/MfluxArtMaker";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { albumSchema, type AlbumFormValues } from "@/lib/schemas";
import { ACE_STEP_KEY_SCALE_OPTIONS, ACE_STEP_TIME_SIGNATURE_OPTIONS } from "@/lib/aceStepSettings";
import {
  api,
  startAlbumJob,
  getAlbumPlanJob,
  PROVIDER_LABEL,
} from "@/lib/api";
import { ACE_STEP_LANGUAGE_OPTIONS } from "@/lib/languages";
import { DEFAULT_LORA_SCALE, emptyLoraSelection, normalizeLoraSelection, type LoraSelection } from "@/lib/lora";
import { DEFAULT_AUDIO_BACKEND, audioBackendLabel, useMlxDitForAudioBackend } from "@/lib/audioBackend";
import { mergeWizardDraft, usePromptMirror, useWizardDraft } from "@/hooks/useWizardDraft";
import { useWizardStore } from "@/store/wizard";
import { useSettingsStore } from "@/store/settings";
import { useJobsStore } from "@/store/jobs";
import { toast } from "@/components/ui/sonner";
import { cn, formatDuration } from "@/lib/utils";

const MODE = "album" as const;
const DEFAULT_ALBUM_TRACK_VARIANTS = 4;
const MAX_ALBUM_TRACK_VARIANTS = 8;
const ALBUM_PORTFOLIO_MODEL_COUNT = 9;

function jobText(value: unknown): string {
  return typeof value === "string" ? value : value == null ? "" : String(value);
}

function jobNumber(value: unknown, fallback = 0): number {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

const SONG_MODELS = [
  ["auto", "Auto (laat de planner kiezen)"],
  ["acestep-v15-xl-sft", "XL SFT"],
  ["acestep-v15-xl-base", "XL Base"],
  ["acestep-v15-xl-turbo", "XL Turbo"],
  ["acestep-v15-sft", "SFT"],
  ["acestep-v15-base", "Base"],
  ["acestep-v15-turbo", "Turbo"],
] as const;

interface AlbumTrack {
  track_number?: number;
  title?: string;
  artist_name?: string;
  duration?: number;
  caption?: string;
  tags: string;
  lyrics?: string;
  bpm?: number;
  key_scale?: string;
  time_signature?: string;
  role?: string;
  planning_status?: string;
  pre_render_repair_status?: string;
  payload_gate_status?: string;
  album_art_prompt?: string;
  album_art_negative_prompt?: string;
  single_art_prompt?: string;
  single_art_negative_prompt?: string;
  video_prompt?: string;
  video_negative_prompt?: string;
  visual_palette?: string;
  camera_motion?: string;
  no_text_policy?: string;
  lyrics_quality?: Record<string, unknown>;
  debug_paths?: Record<string, unknown>;
  use_lora?: boolean;
  lora_adapter_path?: string;
  lora_adapter_name?: string;
  use_lora_trigger?: boolean;
  lora_scale?: number;
  lora_trigger_tag?: string;
  lora_trigger_tags?: string[];
  lora_trigger_source?: string;
  lora_trigger_aliases?: string[];
  lora_trigger_candidates?: string[];
  lora_trigger_applied?: boolean;
  lora_adapters?: LoraSelection["lora_adapters"];
  adapter_model_variant?: string;
  adapter_song_model?: string;
  lora_ignored_reason?: string;
  ignored_lora_adapter_name?: string;
  ignored_lora_adapter_path?: string;
  ignored_lora_adapter_song_model?: string;
  variant_count?: number;
  variant_seeds?: string[];
  track_variant?: number;
  variant_seed?: string | number;
  audios?: Array<Record<string, unknown>>;
  result_id?: string;
  audio_url?: string;
  art?: { url?: string };
  [k: string]: unknown;
}

interface AlbumPlan {
  concept?: string;
  album_title?: string;
  artist_name?: string;
  num_tracks?: number;
  language?: string;
  song_model?: string;
  tracks?: AlbumTrack[];
  album_family_id?: string;
}

function normalizeAlbumTracks(input: unknown, fallbackDuration = 180): AlbumTrack[] {
  if (!Array.isArray(input)) return [];
  return input
    .filter((item): item is Record<string, unknown> => !!item && typeof item === "object")
    .map((item, idx) => {
      const rawDuration = Number(item.duration);
      const rawBpm = Number(item.bpm);
      return {
        ...item,
        track_number: Number(item.track_number) || idx + 1,
        title: String(item.title ?? `Track ${idx + 1}`),
        artist_name: item.artist_name ? String(item.artist_name) : undefined,
        role: item.role ? String(item.role) : undefined,
        duration: Number.isFinite(rawDuration) ? Math.max(30, Math.min(600, rawDuration)) : fallbackDuration,
        caption: item.caption ? String(item.caption) : "",
        tags: item.tags ? String(item.tags) : "",
        lyrics: item.lyrics ? String(item.lyrics) : "",
        bpm: Number.isFinite(rawBpm) ? rawBpm : undefined,
        key_scale: item.key_scale ? String(item.key_scale) : undefined,
        time_signature: item.time_signature ? String(item.time_signature) : undefined,
      } as AlbumTrack;
    });
}

const ALBUM_GENERATE_STRIP_KEYS = new Set([
  "planning_status",
  "planning_error",
  "skip_render",
  "payload_gate_status",
  "payload_gate_passed",
  "payload_gate_blocking_issues",
  "payload_gate_non_blocking",
  "payload_quality_gate",
  "payload_validation",
  "model_results",
  "audios",
  "generated",
  "result_id",
  "active_song_model",
  "audio_url",
  "download_url",
  "song_id",
  "error",
  "debug_paths",
  "agent_debug_dir",
  "agent_rounds",
  "agent_repair_count",
  "repair_actions",
  "lora_ignored_reason",
  "ignored_lora_adapter_name",
  "ignored_lora_adapter_path",
  "ignored_lora_adapter_song_model",
]);

const ALBUM_LEVEL_LORA_KEYS = [
  "use_lora",
  "lora_adapter_path",
  "lora_adapter_name",
  "use_lora_trigger",
  "lora_trigger_tag",
  "lora_trigger_tags",
  "lora_scale",
  "lora_adapters",
  "adapter_model_variant",
  "adapter_song_model",
] as const;

function trackHasExplicitLoraChoice(track: AlbumTrack): boolean {
  if ("use_lora" in track) return true;
  return Boolean(
    track.lora_adapter_path ||
      track.lora_adapter_name ||
      track.lora_trigger_tag ||
      (Array.isArray(track.lora_adapters) && track.lora_adapters.length > 0) ||
      track.adapter_model_variant ||
      track.adapter_song_model ||
      typeof track.lora_scale === "number",
  );
}

function migrateLegacyLoraToTracks(tracks: AlbumTrack[], selection: LoraSelection): AlbumTrack[] {
  if (!selection.use_lora || !selection.lora_adapter_path || tracks.length === 0) return tracks;
  if (tracks.some(trackHasExplicitLoraChoice)) return tracks;
  return tracks.map((track) => ({
    ...track,
    ...selection,
  }));
}

function stripAlbumLevelLoraFields<T extends Record<string, unknown>>(payload: T): T {
  const clean = { ...payload };
  for (const key of ALBUM_LEVEL_LORA_KEYS) {
    delete clean[key];
  }
  return clean;
}

function albumTrackLoraLabel(track: AlbumTrack): string {
  if (track.lora_ignored_reason) {
    return `LoRA genegeerd · ${track.ignored_lora_adapter_name || track.lora_adapter_name || "adapter"}`;
  }
  if (!track.use_lora || !track.lora_adapter_path) return "";
  const count = Array.isArray(track.lora_adapters) ? track.lora_adapters.length : 0;
  return [
    count > 1 ? `${count} LoRAs` : track.lora_adapter_name || "LoRA",
    Array.isArray(track.lora_trigger_tags) && track.lora_trigger_tags.length
      ? track.lora_trigger_tags.join(" + ")
      : track.lora_trigger_tag,
    typeof track.lora_scale === "number" ? `${Math.round(track.lora_scale * 100)}%` : "",
  ].filter(Boolean).join(" · ");
}

function sanitizeAlbumTrackForGenerate(track: AlbumTrack): AlbumTrack {
  const clean = Object.fromEntries(
    Object.entries(track).filter(([key]) => !ALBUM_GENERATE_STRIP_KEYS.has(key)),
  ) as AlbumTrack;
  return {
    ...clean,
    planning_status: "ui_approved",
    skip_render: false,
  };
}

export function AlbumWizard() {
  const navigate = useNavigate();
  const setResult = useWizardStore((s) => s.setResult);
  const lastResult = useWizardStore((s) => s.lastResult[MODE]);
  const warnings = useWizardStore((s) => s.warnings[MODE]) ?? [];
  const storePrompt = useWizardStore((s) => s.prompts[MODE]);
  const draft = useWizardStore((s) => s.drafts[MODE]);
  const plannerProvider = useSettingsStore((s) => s.plannerProvider);
  const plannerModel = useSettingsStore((s) => s.plannerModel);
  const embeddingProvider = useSettingsStore((s) => s.embeddingProvider);
  const embeddingModel = useSettingsStore((s) => s.embeddingModel);
  const albumDefaults = React.useMemo<AlbumFormValues>(
    () => ({
      concept: "",
      num_tracks: 7,
      track_duration: 180,
      track_variants: DEFAULT_ALBUM_TRACK_VARIANTS,
      duration_mode: "ai_per_track",
      album_writer_mode: "per_track_writer_loop",
      language: "en",
      song_model: "acestep-v15-xl-sft",
      audio_backend: DEFAULT_AUDIO_BACKEND,
      song_model_strategy: "single_model_album",
      quality_profile: "chart_master",
      style_profile: "auto",
      custom_tags: "",
      negative_tags: "",
      use_lora: false,
      lora_adapter_path: "",
      lora_adapter_name: "",
      use_lora_trigger: false,
      lora_trigger_tag: "",
      lora_trigger_tags: [],
      lora_scale: DEFAULT_LORA_SCALE,
      lora_adapters: [],
      adapter_model_variant: "",
      adapter_song_model: "",
      auto_song_art: false,
      auto_album_art: false,
      auto_video_clip: false,
      art_prompt: "",
      album_art_prompt: "",
      album_art_negative_prompt: "",
      single_art_prompt: "",
      single_art_negative_prompt: "",
      video_prompt: "",
      video_negative_prompt: "",
      visual_palette: "",
      camera_motion: "",
      no_text_policy: "",
      tracks: [],
    }),
    [],
  );

  const form = useForm<AlbumFormValues>({
    resolver: zodResolver(albumSchema),
    defaultValues: mergeWizardDraft<AlbumFormValues>(albumDefaults, draft),
    mode: "onChange",
  });

  const [step, setStep] = React.useState(0);
  const [aiPromptPending, setAiPromptPending] = React.useState(false);
  const [plan, setPlan] = React.useState<AlbumPlan | null>(null);
  const [jobId, setJobId] = React.useState<string | null>(null);
  const [jobProgress, setJobProgress] = React.useState<number>(0);
  const [jobStatus, setJobStatus] = React.useState<string>("");
  const [jobDetail, setJobDetail] = React.useState<Record<string, unknown> | null>(null);
  const values = form.watch();
  const draftState = useWizardDraft(MODE, form);

  usePromptMirror(form, "concept", storePrompt);

  // ---- Hydration from AI prompt assistant ----
  const hydrate = (payload: Record<string, unknown>) => {
    const next: Partial<AlbumFormValues> = {};
    for (const k of [
      "concept",
      "album_title",
      "artist_name",
      "num_tracks",
      "track_duration",
      "track_variants",
      "duration_mode",
      "album_writer_mode",
      "language",
      "song_model",
      "audio_backend",
      "song_model_strategy",
      "album_mood",
      "vocal_type",
      "genre_prompt",
      "style_profile",
      "custom_tags",
      "negative_tags",
      "use_lora",
      "lora_adapter_path",
      "lora_adapter_name",
      "use_lora_trigger",
      "lora_trigger_tag",
      "lora_scale",
      "adapter_model_variant",
      "adapter_song_model",
      "auto_song_art",
      "auto_album_art",
      "auto_video_clip",
      "art_prompt",
      "album_art_prompt",
      "album_art_negative_prompt",
      "single_art_prompt",
      "single_art_negative_prompt",
      "video_prompt",
      "video_negative_prompt",
      "visual_palette",
      "camera_motion",
      "no_text_policy",
    ] as const) {
      if (k in payload) {
        // @ts-expect-error dynamic
        next[k] = payload[k];
      }
    }
    if (!next.art_prompt && typeof payload.album_art_prompt === "string" && payload.album_art_prompt.trim()) {
      next.art_prompt = payload.album_art_prompt;
    }
    if (!next.video_prompt && typeof payload.video_prompt === "string" && payload.video_prompt.trim()) {
      next.video_prompt = payload.video_prompt;
    }
    if (Array.isArray(payload.tracks)) {
      const legacyLora = normalizeLoraSelection({ ...form.getValues(), ...next });
      const hydratedTracks = normalizeAlbumTracks(payload.tracks, Number(next.track_duration ?? values.track_duration ?? 180));
      next.tracks = migrateLegacyLoraToTracks(hydratedTracks, legacyLora);
      next.num_tracks = next.num_tracks ?? hydratedTracks.length;
      if (legacyLora.use_lora) {
        Object.assign(next, emptyLoraSelection());
      }
    }
    const merged = { ...form.getValues(), ...next };
    form.reset(merged);
    if (Array.isArray(merged.tracks) && merged.tracks.length) {
      setPlan({
        concept: (merged.concept as string) ?? values.concept,
        album_title: merged.album_title as string,
        artist_name: merged.artist_name as string,
        num_tracks: merged.num_tracks,
        language: merged.language as string,
        song_model: merged.song_model as string,
        tracks: merged.tracks as AlbumTrack[],
      });
    }
    draftState.saveNow(merged);
    return merged;
  };

  const setTrackLoraSelection = (trackIndex: number, selection: LoraSelection) => {
    const currentTracks = normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration);
    const next = [...currentTracks];
    if (!next[trackIndex]) return;
    next[trackIndex] = {
      ...next[trackIndex],
      ...selection,
      lora_ignored_reason: "",
      ignored_lora_adapter_name: "",
      ignored_lora_adapter_path: "",
      ignored_lora_adapter_song_model: "",
    };
    updatePlanTracks(next);
  };

  const updatePlanTracks = React.useCallback(
    (tracks: AlbumTrack[]) => {
      const normalized = normalizeAlbumTracks(tracks, Number(form.getValues("track_duration") || 180));
      setPlan((current) => ({
        ...(current ?? {}),
        concept: form.getValues("concept"),
        album_title: form.getValues("album_title"),
        artist_name: form.getValues("artist_name"),
        num_tracks: normalized.length || form.getValues("num_tracks"),
        language: form.getValues("language"),
        song_model: form.getValues("song_model"),
        tracks: normalized,
      }));
      form.setValue("tracks", normalized as never, { shouldDirty: true, shouldValidate: true });
      form.setValue("num_tracks", Math.max(1, normalized.length || form.getValues("num_tracks")), { shouldDirty: true, shouldValidate: true });
      draftState.saveNow({ ...form.getValues(), tracks: normalized, num_tracks: normalized.length || form.getValues("num_tracks") });
    },
    [draftState, form],
  );

  const legacyLoraMigratedRef = React.useRef(false);

  React.useEffect(() => {
    if (legacyLoraMigratedRef.current) return;
    const legacyLora = normalizeLoraSelection(values);
    if (!legacyLora.use_lora) return;
    const currentTracks = normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, Number(values.track_duration || 180));
    if (!currentTracks.length) return;
    legacyLoraMigratedRef.current = true;
    const migratedTracks = migrateLegacyLoraToTracks(currentTracks, legacyLora);
    const empty = emptyLoraSelection();
    for (const key of ALBUM_LEVEL_LORA_KEYS) {
      form.setValue(key as keyof AlbumFormValues, empty[key] as never, { shouldDirty: true, shouldValidate: true });
    }
    updatePlanTracks(migratedTracks);
    draftState.saveNow({
      ...stripAlbumLevelLoraFields(form.getValues()),
      ...empty,
      tracks: migratedTracks,
    });
  }, [draftState, form, plan?.tracks, updatePlanTracks, values, values.track_duration, values.tracks]);

  React.useEffect(() => {
    const draftTracks = normalizeAlbumTracks(values.tracks, Number(values.track_duration || 180));
    if (!draftTracks.length || (plan?.tracks?.length ?? 0) > 0) return;
    setPlan({
      concept: values.concept,
      album_title: values.album_title,
      artist_name: values.artist_name,
      num_tracks: values.num_tracks,
      language: values.language,
      song_model: values.song_model,
      tracks: draftTracks,
    });
  }, [plan?.tracks?.length, values.album_title, values.artist_name, values.concept, values.language, values.num_tracks, values.song_model, values.track_duration, values.tracks]);

  const reviewTracks = React.useMemo(
    () => normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration),
    [plan?.tracks, values.track_duration, values.tracks],
  );
  const albumTrackVariantCount = Math.max(
    1,
    Math.min(MAX_ALBUM_TRACK_VARIANTS, Number(values.track_variants || DEFAULT_ALBUM_TRACK_VARIANTS)),
  );
  const requestedModelAlbumCount = values.song_model_strategy === "all_models_album" ? ALBUM_PORTFOLIO_MODEL_COUNT : 1;
  const requestedAudioRenderCount = Math.max(0, reviewTracks.length) * albumTrackVariantCount * requestedModelAlbumCount;

  const albumCurrentPayload = React.useMemo(
    () => ({
      ...stripAlbumLevelLoraFields(form.getValues()),
      track_variants: albumTrackVariantCount,
      album_render_model_count: requestedModelAlbumCount,
      album_render_audio_count: requestedAudioRenderCount,
      use_mlx_dit: useMlxDitForAudioBackend(values.audio_backend),
      planner_lm_provider: plannerProvider,
      ollama_model: plannerModel || undefined,
      planner_model: plannerModel || undefined,
      embedding_provider: embeddingProvider,
      embedding_lm_provider: embeddingProvider,
      embedding_model: embeddingModel || undefined,
      ace_step_text_encoder: "Qwen3-Embedding-0.6B",
      track_lora_count: reviewTracks.filter((track) => Boolean(track.use_lora && track.lora_adapter_path)).length,
      tracks: reviewTracks,
    }),
    [
      albumTrackVariantCount,
      embeddingModel,
      embeddingProvider,
      form,
      plannerModel,
      plannerProvider,
      requestedAudioRenderCount,
      requestedModelAlbumCount,
      reviewTracks,
      values,
    ],
  );

  // ---- Async generate ----
  const startJob = useMutation({
    mutationFn: () => {
      const tracksForGenerate = reviewTracks.map(sanitizeAlbumTrackForGenerate);
      const body = {
        concept: values.concept,
        album_title: values.album_title,
        artist_name: values.artist_name,
        num_tracks: values.num_tracks,
        track_duration: values.track_duration,
        track_variants: albumTrackVariantCount,
        duration_mode: values.duration_mode,
        album_writer_mode: values.album_writer_mode,
        language: values.language,
        song_model: values.song_model,
        audio_backend: values.audio_backend,
        use_mlx_dit: useMlxDitForAudioBackend(values.audio_backend),
        song_model_strategy: values.song_model_strategy,
        quality_profile: values.quality_profile,
        album_mood: values.album_mood,
        vocal_type: values.vocal_type,
        genre_prompt: values.genre_prompt,
        style_profile: values.style_profile,
        custom_tags: values.custom_tags,
        negative_tags: values.negative_tags,
        auto_song_art: values.auto_song_art,
        auto_album_art: values.auto_album_art,
        auto_video_clip: values.auto_video_clip,
        art_prompt: values.art_prompt,
        album_art_prompt: values.album_art_prompt,
        album_art_negative_prompt: values.album_art_negative_prompt,
        single_art_prompt: values.single_art_prompt,
        single_art_negative_prompt: values.single_art_negative_prompt,
        video_prompt: values.video_prompt,
        video_negative_prompt: values.video_negative_prompt,
        visual_palette: values.visual_palette,
        camera_motion: values.camera_motion,
        no_text_policy: values.no_text_policy,
        planner_lm_provider: plannerProvider,
        ollama_model: plannerModel || undefined,
        planner_model: plannerModel || undefined,
        embedding_provider: embeddingProvider,
        embedding_lm_provider: embeddingProvider,
        embedding_model: embeddingModel || undefined,
        ace_step_text_encoder: "Qwen3-Embedding-0.6B",
        tracks: tracksForGenerate,
        album_generation_mode: "render_existing_tracks",
        render_from_existing_tracks: true,
        skip_album_planning: true,
      };
      return startAlbumJob(body);
    },
    onSuccess: (resp) => {
      if (!resp.success || !resp.job_id) {
        toast.error(resp.error || "Job kon niet starten");
        return;
      }
      setJobId(resp.job_id);
      setJobStatus("queued");
      setJobDetail((resp.job as Record<string, unknown> | undefined) ?? null);
      setStep(4);
    },
    onError: (err: Error) => toast.error(err.message),
  });

  // Poll job status + mirror into global JobTracker
  const addJob = useJobsStore((s) => s.addJob);
  const updateJob = useJobsStore((s) => s.updateJob);

  React.useEffect(() => {
    if (!jobId) return;
    addJob({
      id: jobId,
      kind: "album",
      label: values.album_title || "Album genereren",
      progress: 0,
      status: "queued",
      state: "queued",
      kindLabel: "Album job",
      detailsPath: `/api/album/jobs/${encodeURIComponent(jobId)}`,
      metadata: {
        album_title: values.album_title,
        artist_name: values.artist_name,
        concept: values.concept,
        num_tracks: values.num_tracks,
        track_duration: values.track_duration,
        track_variants: albumTrackVariantCount,
        duration_mode: values.duration_mode,
        planner_provider: plannerProvider,
        planner_model: plannerModel,
        embedding_provider: embeddingProvider,
        embedding_model: embeddingModel,
      },
      startedAt: Date.now(),
    });
    let cancelled = false;
    const poll = async () => {
      try {
        const resp = await getAlbumPlanJob(jobId);
        if (cancelled) return;
        const j = resp.job;
        if (!j) return;
        setJobDetail(j as Record<string, unknown>);
        const state = (j.state || "running").toLowerCase();
        const description = j.status || state;
        setJobStatus(description);
        const p = typeof j.progress === "number" ? j.progress : 0;
        setJobProgress(p);
        updateJob(jobId, {
          progress: p,
          status: description,
          state,
          detailsPath: `/api/album/jobs/${encodeURIComponent(jobId)}`,
          metadata: j as Record<string, unknown>,
          error: j.error,
        });
        if (["complete", "completed", "succeeded", "success"].includes(state) && j.result) {
          setResult(MODE, j.result);
          toast.success("Album klaar.");
          updateJob(jobId, {
            status: "complete",
            state: "complete",
            progress: 100,
            metadata: j as Record<string, unknown>,
          });
          setJobId(null);
          return;
        }
        if (state === "error" || state === "failed") {
          toast.error(j.error || "Album-job mislukt");
          updateJob(jobId, {
            status: "error",
            state,
            error: j.error,
            metadata: j as Record<string, unknown>,
          });
          setJobId(null);
          return;
        }
        setTimeout(poll, 2500);
      } catch (e) {
        if (!cancelled) {
          const message = (e as Error).message;
          updateJob(jobId, {
            status: "Albumstatus tijdelijk niet bereikbaar",
            state: "running",
            error: message,
          });
          setTimeout(poll, 4000);
        }
      }
    };
    poll();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  const albumFamilyId =
    (lastResult?.album_family_id as string | undefined) ||
    plan?.album_family_id;
  const resultTracks = (lastResult?.tracks as AlbumTrack[] | undefined) ?? [];
  const albumArt = lastResult?.album_art as { url?: string } | undefined;

  const steps: WizardStepDef[] = [
    {
      key: "ai",
      title: "Album-concept",
      description:
        "Beschrijf het album. AI plant tracks, taal, BPM-strategie en kwaliteitsprofiel.",
      isValid: (values.concept ?? "").trim().length >= 8 && !aiPromptPending,
      render: () => (
        <AIPromptStep
          mode="album"
          placeholder="Bijv. '7-track concept-album over een verlaten kustdorp, melancholische post-rock met momenten van hoop, NL/EN mix'"
          examples={[
            "5 tracks lo-fi hip-hop voor een rustige werkmiddag",
            "10-track epische orkestrale soundtrack voor een sci-fi film",
            "EP van 4 tracks afrobeat × jazz, 100-115 bpm",
          ]}
          currentPayload={albumCurrentPayload}
          onPendingChange={setAiPromptPending}
          onHydrated={(payload) => {
            const merged = hydrate(payload);
            const c =
              (payload.concept as string | undefined) ??
              (payload.user_prompt as string | undefined) ??
              form.getValues("concept");
            if (c) {
              const withConcept = { ...merged, concept: c };
              form.reset(withConcept);
              draftState.saveNow(withConcept);
            }
          }}
          onManualApply={() => {
            setStep(1);
          }}
        />
      ),
    },
    {
      key: "identity",
      title: "Album-identiteit",
      isValid: true,
      render: () => (
        <div className="space-y-4">
          <FieldGroup title="Naam">
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1.5">
                <Label>Album-titel</Label>
                <Input {...form.register("album_title")} />
              </div>
              <div className="space-y-1.5">
                <Label>Artiest</Label>
                <Input {...form.register("artist_name")} />
              </div>
            </div>
          </FieldGroup>
          <FieldGroup title="Aantal tracks & duur">
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-3">
                <div className="flex items-baseline justify-between">
                  <Label>Tracks</Label>
                  <span className="font-mono text-sm">{values.num_tracks}</span>
                </div>
                <Controller
                  control={form.control}
                  name="num_tracks"
                  render={({ field }) => (
                    <Slider value={[field.value]} min={1} max={20} step={1} onValueChange={(v) => field.onChange(v[0])} />
                  )}
                />
              </div>
              <div className="space-y-3">
                <div className="flex items-baseline justify-between">
                  <Label>Gemiddelde/fallback duur</Label>
                  <span className="font-mono text-sm">{formatDuration(values.track_duration)}</span>
                </div>
                <Controller
                  control={form.control}
                  name="track_duration"
                  render={({ field }) => (
                    <Slider value={[field.value]} min={30} max={600} step={5} onValueChange={(v) => field.onChange(v[0])} />
                  )}
                />
              </div>
            </div>
            <div className="space-y-1.5">
              <Label>Duurbeleid</Label>
              <Controller
                control={form.control}
                name="duration_mode"
                render={({ field }) => (
                  <Select value={field.value} onValueChange={field.onChange}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="ai_per_track">AI per track</SelectItem>
                      <SelectItem value="fixed">Vaste duur voor alle tracks</SelectItem>
                    </SelectContent>
                  </Select>
                )}
              />
              <p className="text-xs text-muted-foreground">
                AI per track houdt intro’s, interludes en outro’s korter en laat singles/full songs langer ademen.
              </p>
            </div>
          </FieldGroup>
          <FieldGroup title="Stijl">
            <AudioStyleSelector
              value={values.style_profile}
              onChange={(value) => form.setValue("style_profile", value, { shouldValidate: true })}
            />
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1.5">
                <Label>Sfeer</Label>
                <Input placeholder="bijv. melancholic, hopeful" {...form.register("album_mood")} />
              </div>
              <div className="space-y-1.5">
                <Label>Vocals</Label>
                <Input placeholder="bijv. female lead, mixed choir" {...form.register("vocal_type")} />
              </div>
            </div>
            <div className="space-y-1.5">
              <Label>Genre-richting</Label>
              <Textarea rows={2} {...form.register("genre_prompt")} />
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1.5">
                <Label>Album-tags</Label>
                <Textarea rows={2} {...form.register("custom_tags")} />
              </div>
              <div className="space-y-1.5">
                <Label>Negative tags</Label>
                <Textarea rows={2} {...form.register("negative_tags")} />
              </div>
            </div>
          </FieldGroup>
        </div>
      ),
    },
    {
      key: "tracks",
      title: "Tracklist",
      description: "AI heeft een tracklist voorgesteld. Bewerk titels, ruil ze om of voeg er een toe.",
      isValid: true,
      render: () => (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">
              {normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration).length} tracks gepland
            </p>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                const currentTracks = normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration);
                const next = [...currentTracks];
                next.push({
                  track_number: next.length + 1,
                  title: `Track ${next.length + 1}`,
                  duration: values.track_duration,
                  caption: "",
                  tags: "",
                  lyrics: "",
                });
                updatePlanTracks(next);
              }}
              className="gap-1.5"
            >
              <Plus className="size-3.5" /> Track toevoegen
            </Button>
          </div>
          <AnimatePresence initial={false}>
            {normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration).map((t, idx) => (
              <motion.div
                key={`${t.title}-${idx}`}
                layout
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                className="rounded-xl border bg-card/40 p-3"
              >
                <div className="flex items-start gap-3">
                  <div className="flex size-8 shrink-0 items-center justify-center rounded-md bg-primary/15 font-mono text-xs text-primary">
                    {String(idx + 1).padStart(2, "0")}
                  </div>
                  <div className="min-w-0 flex-1 space-y-2">
                    <Input
                      value={t.title ?? ""}
                      onChange={(e) => {
                        const next = normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration);
                        next[idx] = { ...next[idx], title: e.target.value };
                        updatePlanTracks(next);
                      }}
                      placeholder="Tracktitel"
                    />
                    <Textarea
                      rows={2}
                      value={t.caption ?? ""}
                      onChange={(e) => {
                        const next = normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration);
                        next[idx] = { ...next[idx], caption: e.target.value };
                        updatePlanTracks(next);
                      }}
                      placeholder="Caption / vibe"
                    />
                    <div className="grid gap-2 sm:grid-cols-4">
                      <div className="space-y-1">
                        <Label className="text-xs">Rol</Label>
                        <Input
                          value={t.role ?? ""}
                          onChange={(e) => {
                            const next = normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration);
                            next[idx] = { ...next[idx], role: e.target.value };
                            updatePlanTracks(next);
                          }}
                          placeholder="single"
                        />
                      </div>
                      <div className="space-y-1">
                        <Label className="text-xs">Duur</Label>
                        <Input
                          type="number"
                          min={30}
                          max={600}
                          value={t.duration ?? values.track_duration}
                          onChange={(e) => {
                            const next = normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration);
                            next[idx] = {
                              ...next[idx],
                              duration: Number(e.target.value) || values.track_duration,
                              duration_locked: true,
                              duration_source: "manual_track",
                            };
                            updatePlanTracks(next);
                          }}
                        />
                      </div>
                      <div className="space-y-1">
                        <Label className="text-xs">BPM</Label>
                        <Input
                          type="number"
                          value={t.bpm ?? ""}
                          onChange={(e) => {
                            const next = normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration);
                            next[idx] = { ...next[idx], bpm: Number(e.target.value) || undefined };
                            updatePlanTracks(next);
                          }}
                          placeholder="92"
                        />
                      </div>
                      <div className="space-y-1">
                        <Label className="text-xs">Key</Label>
                        <Select
                          value={t.key_scale || "auto"}
                          onValueChange={(value) => {
                            const next = normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration);
                            next[idx] = { ...next[idx], key_scale: value === "auto" ? undefined : value };
                            updatePlanTracks(next);
                          }}
                        >
                          <SelectTrigger><SelectValue /></SelectTrigger>
                          <SelectContent>
                            {ACE_STEP_KEY_SCALE_OPTIONS.map((value) => (
                              <SelectItem key={value} value={value}>{value === "auto" ? "Auto" : value}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                    <div className="space-y-1">
                      <Label className="text-xs">Time signature</Label>
                      <Select
                        value={t.time_signature || "auto"}
                        onValueChange={(value) => {
                          const next = normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration);
                          next[idx] = { ...next[idx], time_signature: value === "auto" ? undefined : value };
                          updatePlanTracks(next);
                        }}
                      >
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          {ACE_STEP_TIME_SIGNATURE_OPTIONS.map(([value, label]) => (
                            <SelectItem key={value || "auto"} value={value || "auto"}>{label}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <details className="rounded-lg border bg-background/40 p-2">
                      <summary className="cursor-pointer text-xs font-medium text-muted-foreground">
                        Stijl & ACE-Step tags
                      </summary>
                      <div className="mt-2 grid gap-2 sm:grid-cols-2">
                        <div className="space-y-1">
                          <Label className="text-xs">Genre / stijlprofiel</Label>
                          <Input
                            value={String(t.style_profile ?? t.genre_profile ?? t.genre ?? "")}
                            onChange={(e) => {
                              const next = normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration);
                              next[idx] = { ...next[idx], style_profile: e.target.value };
                              updatePlanTracks(next);
                            }}
                          />
                        </div>
                        <div className="space-y-1">
                          <Label className="text-xs">Genre-richting</Label>
                          <Input
                            value={String(t.genre_direction ?? t.genre ?? "")}
                            onChange={(e) => {
                              const next = normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration);
                              next[idx] = { ...next[idx], genre_direction: e.target.value, genre: e.target.value };
                              updatePlanTracks(next);
                            }}
                          />
                        </div>
                      </div>
                      <div className="mt-2 grid gap-2 sm:grid-cols-2">
                        <div className="space-y-1">
                          <Label className="text-xs">Caption tags</Label>
                          <Textarea
                            rows={2}
                            value={String(t.caption_tags ?? t.tags ?? t.caption ?? "")}
                            onChange={(e) => {
                              const next = normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration);
                              next[idx] = { ...next[idx], caption_tags: e.target.value, tags: e.target.value, caption: e.target.value };
                              updatePlanTracks(next);
                            }}
                          />
                        </div>
                        <div className="space-y-1">
                          <Label className="text-xs">Album-tags / negative tags</Label>
                          <Textarea
                            rows={2}
                            value={[
                              String(t.album_tags ?? ""),
                              String(t.negative_tags ?? t.negative_control ?? ""),
                            ].filter(Boolean).join("\n---\n")}
                            onChange={(e) => {
                              const [albumTags = "", negativeTags = ""] = e.target.value.split(/\n---\n/);
                              const next = normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration);
                              next[idx] = { ...next[idx], album_tags: albumTags, negative_tags: negativeTags, negative_control: negativeTags };
                              updatePlanTracks(next);
                            }}
                          />
                        </div>
                      </div>
                    </details>
                    <details className="rounded-lg border bg-background/40 p-2">
                      <summary className="cursor-pointer text-xs font-medium text-muted-foreground">
                        Track LoRA
                        {albumTrackLoraLabel(t) ? <Badge variant="outline" className="ml-2">{albumTrackLoraLabel(t)}</Badge> : null}
                      </summary>
                      <div className="mt-3">
                        <LoraSelector value={t} onChange={(selection) => setTrackLoraSelection(idx, selection)} />
                        {t.lora_ignored_reason ? (
                          <p className="mt-2 rounded-md border border-amber-400/30 bg-amber-400/10 p-2 text-xs text-amber-100">
                            {t.lora_ignored_reason}
                          </p>
                        ) : null}
                      </div>
                    </details>
                    <details className="rounded-lg border bg-background/40 p-2">
                      <summary className="flex cursor-pointer items-center gap-2 text-xs font-medium text-muted-foreground">
                        <Sparkles className="size-3.5" />
                        Art & video prompts
                        {t.single_art_prompt ? <Badge variant="outline" className="ml-1">single art</Badge> : null}
                        {t.video_prompt ? <Badge variant="outline" className="ml-1">video</Badge> : null}
                      </summary>
                      <div className="mt-2 grid gap-2 sm:grid-cols-2">
                        <div className="space-y-1">
                          <Label className="text-xs">Single art prompt</Label>
                          <Textarea
                            rows={3}
                            value={String(t.single_art_prompt ?? "")}
                            onChange={(e) => {
                              const next = normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration);
                              next[idx] = { ...next[idx], single_art_prompt: e.target.value };
                              updatePlanTracks(next);
                            }}
                          />
                        </div>
                        <div className="space-y-1">
                          <Label className="text-xs">Single negative prompt</Label>
                          <Textarea
                            rows={3}
                            value={String(t.single_art_negative_prompt ?? "")}
                            onChange={(e) => {
                              const next = normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration);
                              next[idx] = { ...next[idx], single_art_negative_prompt: e.target.value };
                              updatePlanTracks(next);
                            }}
                          />
                        </div>
                      </div>
                      <div className="mt-2 grid gap-2 sm:grid-cols-2">
                        <div className="space-y-1">
                          <Label className="flex items-center gap-1 text-xs">
                            <Video className="size-3" /> Video prompt
                          </Label>
                          <Textarea
                            rows={3}
                            value={String(t.video_prompt ?? "")}
                            onChange={(e) => {
                              const next = normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration);
                              next[idx] = { ...next[idx], video_prompt: e.target.value };
                              updatePlanTracks(next);
                            }}
                          />
                        </div>
                        <div className="space-y-1">
                          <Label className="text-xs">Video negative prompt</Label>
                          <Textarea
                            rows={3}
                            value={String(t.video_negative_prompt ?? "")}
                            onChange={(e) => {
                              const next = normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration);
                              next[idx] = { ...next[idx], video_negative_prompt: e.target.value };
                              updatePlanTracks(next);
                            }}
                          />
                        </div>
                      </div>
                      <div className="mt-2 grid gap-2 sm:grid-cols-3">
                        <div className="space-y-1">
                          <Label className="text-xs">Palette</Label>
                          <Input
                            value={String(t.visual_palette ?? "")}
                            onChange={(e) => {
                              const next = normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration);
                              next[idx] = { ...next[idx], visual_palette: e.target.value };
                              updatePlanTracks(next);
                            }}
                          />
                        </div>
                        <div className="space-y-1">
                          <Label className="text-xs">Camera motion</Label>
                          <Input
                            value={String(t.camera_motion ?? "")}
                            onChange={(e) => {
                              const next = normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration);
                              next[idx] = { ...next[idx], camera_motion: e.target.value };
                              updatePlanTracks(next);
                            }}
                          />
                        </div>
                        <div className="space-y-1">
                          <Label className="text-xs">No-text policy</Label>
                          <Input
                            value={String(t.no_text_policy ?? "")}
                            onChange={(e) => {
                              const next = normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration);
                              next[idx] = { ...next[idx], no_text_policy: e.target.value };
                              updatePlanTracks(next);
                            }}
                          />
                        </div>
                      </div>
                    </details>
                    <details className="rounded-lg border bg-background/40 p-2">
                      <summary className="cursor-pointer text-xs font-medium text-muted-foreground">
                        Lyrics-preview / bewerken
                      </summary>
                      <Textarea
                        className="mt-2"
                        rows={6}
                        value={t.lyrics ?? ""}
                        onChange={(e) => {
                          const next = normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration);
                          next[idx] = { ...next[idx], lyrics: e.target.value };
                          updatePlanTracks(next);
                        }}
                        placeholder="[Verse]..."
                      />
                    </details>
                    <div className="flex flex-wrap gap-1.5">
                      {typeof t.bpm === "number" && (
                        <Badge variant="muted" className="text-[10px]">{t.bpm} bpm</Badge>
                      )}
                      {t.key_scale && (
                        <Badge variant="muted" className="text-[10px]">{t.key_scale}</Badge>
                      )}
                      {t.role && (
                        <Badge variant="outline" className="text-[10px]">{t.role}</Badge>
                      )}
                      {typeof t.duration === "number" && (
                        <Badge variant="muted" className="text-[10px]">{formatDuration(t.duration)}</Badge>
                      )}
                      <Badge variant="muted" className="text-[10px]">
                        {trackQuality(t).wordCount} woorden
                      </Badge>
                      <Badge variant="muted" className="text-[10px]">
                        {trackQuality(t).lineCount} regels · {trackQuality(t).sectionCount} secties
                      </Badge>
                      <Badge variant={trackQuality(t).hookCount > 0 ? "muted" : "destructive"} className="text-[10px]">
                        hooks {trackQuality(t).hookCount}
                      </Badge>
                      {(String(t.planning_status || "").toLowerCase() === "failed" || String(t.payload_gate_status || "").toLowerCase() === "planning_failed") && (
                        <Badge variant="outline" className="text-[10px]">
                          needs repair
                        </Badge>
                      )}
                      {Object.keys(trackQuality(t).rapBarCounts).length > 0 && (
                        <Badge variant="outline" className="text-[10px]">
                          rap bars {Object.values(trackQuality(t).rapBarCounts).map((count) => String(count)).join("/")}
                        </Badge>
                      )}
                      {trackQuality(t).gateStatus && (
                        <Badge variant={trackQuality(t).gateStatus === "pass" ? "default" : "destructive"} className="text-[10px]">
                          gate {trackQuality(t).gateStatus}
                        </Badge>
                      )}
                      {albumTrackLoraLabel(t) && (
                        <Badge variant={t.lora_ignored_reason ? "destructive" : "outline"} className="text-[10px]">
                          {albumTrackLoraLabel(t)}
                        </Badge>
                      )}
                    </div>
                    {t.debug_paths && Object.keys(t.debug_paths).length > 0 && (
                      <details className="rounded-lg border bg-background/40 p-2">
                        <summary className="cursor-pointer text-xs font-medium text-muted-foreground">
                          Debug-output
                        </summary>
                        <pre className="mt-2 max-h-40 overflow-auto rounded-md bg-muted p-2 text-[10px]">
                          {JSON.stringify(t.debug_paths, null, 2)}
                        </pre>
                      </details>
                    )}
                  </div>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={() => {
                      const next = normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration).filter((_, i) => i !== idx);
                      updatePlanTracks(next);
                    }}
                    title="Verwijder track"
                  >
                    <Trash2 className="size-3.5" />
                  </Button>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          {normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration).length === 0 && (
            <div className="rounded-xl border border-dashed bg-card/20 p-8 text-center text-sm text-muted-foreground">
              Geen tracklist gepland. Ga terug naar stap 0 en klik <em>Vul met AI</em>.
            </div>
          )}
        </div>
      ),
    },
    {
      key: "render",
      title: "Render-instellingen",
      isValid: true,
      render: () => (
        <div className="space-y-4">
          <FieldGroup title="Model strategie">
            <div className="grid gap-3 sm:grid-cols-3">
              <div className="space-y-1.5">
                <Label>Song model</Label>
                <Controller
                  control={form.control}
                  name="song_model"
                  render={({ field }) => (
                    <Select value={field.value} onValueChange={field.onChange}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {SONG_MODELS.map(([id, label]) => (
                          <SelectItem key={id} value={id}>{label}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>
              <div className="space-y-1.5">
                <Label>Strategie</Label>
                <Controller
                  control={form.control}
                  name="song_model_strategy"
                  render={({ field }) => (
                    <Select value={field.value} onValueChange={field.onChange}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="single_model_album">Eén model</SelectItem>
                        <SelectItem value="all_models_album">Portfolio (alle modellen)</SelectItem>
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>
              <AudioBackendSelector
                value={values.audio_backend}
                onChange={(value) => form.setValue("audio_backend", value, { shouldValidate: true })}
              />
            </div>
          </FieldGroup>
          <FieldGroup title="Versies">
            <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_minmax(180px,240px)]">
              <div className="space-y-1.5">
                <Label>Versies per track</Label>
                <Controller
                  control={form.control}
                  name="track_variants"
                  render={({ field }) => (
                    <Input
                      type="number"
                      min={1}
                      max={MAX_ALBUM_TRACK_VARIANTS}
                      step={1}
                      value={field.value ?? DEFAULT_ALBUM_TRACK_VARIANTS}
                      onChange={(event) => {
                        const parsed = Number(event.target.value);
                        const next = Number.isFinite(parsed)
                          ? Math.max(1, Math.min(MAX_ALBUM_TRACK_VARIANTS, Math.round(parsed)))
                          : DEFAULT_ALBUM_TRACK_VARIANTS;
                        field.onChange(next);
                      }}
                    />
                  )}
                />
              </div>
              <div className="rounded-lg border border-border/60 bg-background/40 p-3">
                <p className="text-[11px] uppercase tracking-wider text-muted-foreground">Totaal</p>
                <p className="mt-1 text-sm font-medium">{requestedAudioRenderCount} audio renders</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  {reviewTracks.length} tracks × {albumTrackVariantCount} versies × {requestedModelAlbumCount} model{requestedModelAlbumCount === 1 ? "" : "len"}
                </p>
              </div>
            </div>
            <p className="text-xs text-muted-foreground">
              Elke versie gebruikt dezelfde lyrics, caption, metadata en LoRA; alleen de ACE-Step seed verandert.
            </p>
          </FieldGroup>
          <FieldGroup title="Kwaliteit">
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1.5">
                <Label>Quality profile</Label>
                <Controller
                  control={form.control}
                  name="quality_profile"
                  render={({ field }) => (
                    <Select value={field.value} onValueChange={field.onChange}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="draft">Laag</SelectItem>
                        <SelectItem value="standard">Middel</SelectItem>
                        <SelectItem value="chart_master">Hoog</SelectItem>
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>
              <div className="space-y-1.5">
                <Label>Taal</Label>
                <Controller
                  control={form.control}
                  name="language"
                  render={({ field }) => (
                    <Select value={field.value || "unknown"} onValueChange={field.onChange}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {ACE_STEP_LANGUAGE_OPTIONS.map(([code, label]) => (
                          <SelectItem key={code} value={code}>{label}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>
            </div>
          </FieldGroup>
          <FieldGroup
            title="AI Memory / RAG embeddings"
            description="Gebruikt door CrewAI album-memory en retrieval. De ACE-Step audio text encoder blijft vast."
          >
            <div className="grid gap-3 sm:grid-cols-3">
              <div className="rounded-lg border border-border/60 bg-background/40 p-3">
                <p className="text-[11px] uppercase tracking-wider text-muted-foreground">Provider</p>
                <p className="mt-1 text-sm font-medium">{PROVIDER_LABEL[embeddingProvider]}</p>
              </div>
              <div className="rounded-lg border border-border/60 bg-background/40 p-3">
                <p className="text-[11px] uppercase tracking-wider text-muted-foreground">Embedding model</p>
                <p className="mt-1 truncate text-sm font-medium">{embeddingModel || "Settings default"}</p>
              </div>
              <div className="rounded-lg border border-border/60 bg-background/40 p-3">
                <p className="text-[11px] uppercase tracking-wider text-muted-foreground">ACE-Step encoder</p>
                <p className="mt-1 text-sm font-medium">Qwen3-Embedding-0.6B</p>
              </div>
            </div>
            <p className="text-xs text-muted-foreground">
              Pas deze keuze aan in de AI Fill-kaart of in Settings; albumjobs sturen dezelfde provider en model mee naar CrewAI.
            </p>
          </FieldGroup>
          <AutomationFields control={form.control} register={form.register} values={values} albumContext />
        </div>
      ),
    },
    {
      key: "review",
      title: "Review & start render",
      isValid: form.formState.isValid,
      render: () => (
        <div className="space-y-4">
          <ReviewStep
            payload={{
              ...albumCurrentPayload,
            }}
            warnings={warnings}
            primaryFields={[
              { key: "album_title", label: "Album titel" },
              { key: "artist_name", label: "Artiest" },
              { key: "num_tracks", label: "Tracks" },
              { key: "track_variants", label: "Versies per track" },
              { key: "album_render_audio_count", label: "Totaal renders", format: (v) => `${Number(v) || 0} audio renders` },
              { key: "duration_mode", label: "Duurbeleid", format: (v) => v === "fixed" ? "Vaste duur" : "AI per track" },
              { key: "album_writer_mode", label: "AI writer", format: () => "Per-track loop" },
              { key: "track_duration", label: "Fallback duur", format: (v) => formatDuration(Number(v) || 0) },
              { key: "song_model", label: "Model" },
              { key: "audio_backend", label: "Backend", format: audioBackendLabel },
              { key: "planner_model", label: "AI planner", format: (v) => `${PROVIDER_LABEL[plannerProvider]} · ${String(v || "—")}` },
              { key: "embedding_model", label: "AI Memory", format: (v) => `${PROVIDER_LABEL[embeddingProvider]} · ${String(v || "—")}` },
              { key: "ace_step_text_encoder", label: "ACE-Step encoder" },
              { key: "track_lora_count", label: "Track-LoRAs", format: (v) => `${Number(v) || 0} track(s)` },
              { key: "song_model_strategy", label: "Strategie" },
              { key: "language", label: "Taal" },
              { key: "quality_profile", label: "Kwaliteit" },
            ]}
          />
        </div>
      ),
    },
    {
      key: "rendering",
      title: "Album genereren",
      description:
        jobId ? "ACE-Step is je album aan het renderen. Dit duurt enkele minuten per track." : "Klaar met genereren.",
      isValid: !jobId,
      hidden: !jobId && !lastResult,
      render: () => (
        <div className="space-y-4">
          <div className="rounded-xl border border-primary/30 bg-primary/5 p-5 text-sm">
            <div className="flex items-center gap-3">
              {jobId ? (
                <Loader2 className="size-5 animate-spin text-primary" />
              ) : (
                <Disc3 className="size-5 text-primary" />
              )}
              <div className="flex-1">
                <p className="font-medium">{jobId ? "Renderen…" : "Klaar"}</p>
                <p className="text-xs text-muted-foreground">{jobStatus || "—"}</p>
              </div>
              <span className="font-mono text-sm tabular-nums">{jobProgress}%</span>
            </div>
            <Progress value={jobProgress} className="mt-3" />
            {jobDetail && (
              <div className="mt-4 grid gap-2 text-xs text-muted-foreground sm:grid-cols-2 lg:grid-cols-4">
                <div>
                  Taak{" "}
                  <span className="font-medium text-foreground">
                    {jobText(jobDetail.current_task) || jobText(jobDetail.stage) || "—"}
                  </span>
                </div>
                <div>
                  Track{" "}
                  <span className="font-medium text-foreground">
                    {jobText(jobDetail.current_track) || "0"}/{jobText(jobDetail.total_tracks) || "?"}
                  </span>
                </div>
                <div>
                  Klaar{" "}
                  <span className="font-medium text-foreground">
                    {jobText(jobDetail.completed_tracks) || "0"}
                  </span>{" "}
                  · Nog{" "}
                  <span className="font-medium text-foreground">
                    {jobText(jobDetail.remaining_tracks) || "?"}
                  </span>
                </div>
                <div>
                  {Boolean(jobDetail.waiting_on_llm) ? (
                    <>
                      Wacht op{" "}
                      <span className="font-medium text-foreground">
                        {jobText(jobDetail.llm_provider) || "LLM"} {Math.round(jobNumber(jobDetail.llm_wait_elapsed_s))}s
                      </span>
                    </>
                  ) : (
                    <>
                      Agent{" "}
                      <span className="font-medium text-foreground">
                        {jobText(jobDetail.current_agent) || "—"}
                      </span>
                    </>
                  )}
                </div>
              </div>
            )}
            {jobDetail && Array.isArray(jobDetail.logs) && jobDetail.logs.length > 0 && (
              <div className="mt-3 rounded-md border border-border/50 bg-background/60 p-2 text-xs text-muted-foreground">
                {String(jobDetail.logs[jobDetail.logs.length - 1])}
              </div>
            )}
          </div>
        </div>
      ),
    },
    {
      key: "result",
      title: "Album",
      isValid: true,
      hidden: !lastResult,
      render: () => {
        if (!lastResult) return null;
        return (
          <div className="space-y-4">
            {albumArt?.url && (
              <motion.div
                initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3 }}
                className="flex items-center gap-4 rounded-2xl border bg-card/40 p-4"
              >
                <img
                  src={albumArt.url}
                  alt="cover"
                  className="size-32 rounded-lg object-cover shadow-xl"
                />
                <div className="space-y-1">
                  <p className="font-display text-2xl font-semibold">
                    {values.album_title || (lastResult.album_title as string | undefined) || "Untitled"}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {values.artist_name || (lastResult.artist_name as string)}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {resultTracks.length} tracks · {values.song_model_strategy}
                  </p>
                </div>
              </motion.div>
            )}
            <MfluxArtMaker
              title={values.album_title || (lastResult.album_title as string | undefined)}
              artist={values.artist_name || (lastResult.artist_name as string | undefined)}
              context={values.concept || (lastResult.concept as string | undefined)}
              targetType="album_family"
              targetId={albumFamilyId || (lastResult.album_id as string | undefined)}
            />
            <div className="space-y-2">
              {resultTracks.map((t, i) => {
                const trackAudios = Array.isArray(t.audios) ? t.audios : [];
                const audioItems = trackAudios.length
                  ? trackAudios
                  : t.audio_url
                    ? [{
                        audio_url: t.audio_url,
                        result_id: t.result_id,
                        song_id: t.song_id,
                        track_variant: t.track_variant || 1,
                        variant_seed: t.variant_seed,
                      }]
                    : [];
                return (
                  <motion.div
                    key={t.result_id || i}
                    initial={{ opacity: 0, x: -6 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.04 }}
                    className="space-y-2 rounded-xl border bg-card/30 p-3"
                  >
                    <div className="flex items-center gap-3">
                      <span className="font-mono text-xs text-muted-foreground">
                        {String(i + 1).padStart(2, "0")}
                      </span>
                      <div className="min-w-0 flex-1">
                        <p className="truncate font-medium">{t.title || "Track"}</p>
                        <p className="truncate text-xs text-muted-foreground">{t.caption}</p>
                      </div>
                      {typeof t.duration === "number" && (
                        <span className="font-mono text-xs text-muted-foreground">
                          {formatDuration(t.duration)}
                        </span>
                      )}
                    </div>
                    {audioItems.length > 0 && (
                      <div className="space-y-3">
                        {audioItems.map((audio, audioIndex) => {
                          const audioUrl = String(audio.audio_url || audio.library_url || "");
                          const variantNumber = Number(audio.track_variant || audioIndex + 1);
                          const variantSeed = audio.variant_seed || audio.seed;
                          if (!audioUrl) return null;
                          return (
                            <div key={`${audioUrl}-${audioIndex}`} className="space-y-2 rounded-lg border border-border/50 bg-background/40 p-2">
                              <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                                <Badge variant="outline">v{variantNumber}</Badge>
                                {variantSeed ? <Badge variant="muted">seed {String(variantSeed)}</Badge> : null}
                              </div>
                              <WaveformPlayer
                                src={audioUrl}
                                title={`${t.title || "Track"} v${variantNumber}`}
                                artist={values.artist_name}
                              />
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => navigate("/wizard/video", {
                                  state: {
                                    audio_url: audioUrl,
                                    title: `${t.title || "Track"} v${variantNumber}`,
                                    artist_name: values.artist_name,
                                    prompt: String(t.caption || values.concept || t.title || ""),
                                    target_type: "song",
                                    target_id: audio.result_id || audio.song_id || `${lastResult.album_id || albumFamilyId || "album"}:track:${i + 1}:v${variantNumber}`,
                                  },
                                })}
                                className="gap-2"
                              >
                                <Video className="size-3.5" />
                                Create video
                              </Button>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </motion.div>
                );
              })}
            </div>
            <div className="flex flex-wrap items-center gap-2 pt-2">
              {(lastResult.album_id || albumFamilyId) && (
                <Button asChild variant="default" className="gap-2">
                  <a
                    href={
                      albumFamilyId
                        ? `/api/album-families/${albumFamilyId}/download`
                        : `/api/albums/${lastResult.album_id}/download`
                    }
                    download
                  >
                    <Music4 className="size-4" /> Download album ZIP
                  </a>
                </Button>
              )}
              <Button variant="outline" onClick={() => navigate("/library")} className="gap-2">
                <Music4 className="size-4" /> Open library
              </Button>
              <Button variant="ghost" onClick={() => { form.reset(albumDefaults); draftState.clear(); setStep(0); setPlan(null); }}>Nog een album</Button>
            </div>
          </div>
        );
      },
    },
  ];

  return (
    <WizardShell
      title="Album wizard"
      subtitle="Concept → tracklist → render. Albumcoherentie via planner-agents."
      steps={steps}
      step={step}
      onStepChange={setStep}
      onFinish={() => startJob.mutate()}
      isFinishing={startJob.isPending || !!jobId}
      finishLabel={jobId ? "Renderen…" : startJob.isPending ? "Job start…" : "Genereer album"}
    />
  );
}

function trackQuality(track: AlbumTrack) {
  const quality = (track.lyrics_quality && typeof track.lyrics_quality === "object" ? track.lyrics_quality : {}) as Record<string, unknown>;
  const lyrics = String(track.lyrics ?? "");
  const lyricLines = lyrics
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !/^\[[^\]]+\]$/.test(line));
  const sectionCount = (lyrics.match(/\[[^\]]+\]/g) ?? []).length;
  const fallbackWords = lyrics.split(/\s+/).filter(Boolean).length;
  const wordCount = Number(quality.word_count ?? fallbackWords);
  const lineCount = Number(quality.line_count ?? lyricLines.length);
  const hookCount = Number(quality.hook_count ?? (lyrics.match(/\[(?:[^\]]*(?:chorus|hook|refrain)[^\]]*)\]/gi) ?? []).length);
  const rapBarCounts = (
    quality.rap_bar_counts && typeof quality.rap_bar_counts === "object"
      ? quality.rap_bar_counts
      : {}
  ) as Record<string, unknown>;
  const rawGateStatus = String(track.payload_gate_status || quality.gate_status || "");
  const gateStatus = rawGateStatus === "planning_failed" ? "needs_repair" : rawGateStatus;
  return {
    wordCount,
    lineCount,
    sectionCount: Number(quality.section_count ?? (sectionCount || 0)),
    hookCount,
    rapBarCounts,
    gateStatus,
    targetWords: Number(quality.target_words || 0),
    minWords: Number(quality.min_words || 0),
  };
}
