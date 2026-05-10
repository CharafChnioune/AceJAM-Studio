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
import {
  api,
  startAlbumPlanJob,
  getAlbumPlanJob,
  PROVIDER_LABEL,
} from "@/lib/api";
import { DEFAULT_LORA_SCALE, normalizeLoraSelection, type LoraSelection } from "@/lib/lora";
import { audioBackendLabel, useMlxDitForAudioBackend } from "@/lib/audioBackend";
import { mergeWizardDraft, usePromptMirror, useWizardDraft } from "@/hooks/useWizardDraft";
import { useWizardStore } from "@/store/wizard";
import { useSettingsStore } from "@/store/settings";
import { useJobsStore } from "@/store/jobs";
import { toast } from "@/components/ui/sonner";
import { cn, formatDuration } from "@/lib/utils";

const MODE = "album" as const;

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
  payload_gate_status?: string;
  lyrics_quality?: Record<string, unknown>;
  debug_paths?: Record<string, unknown>;
  lora_adapter_name?: string;
  lora_scale?: number;
  lora_trigger_tag?: string;
  lora_trigger_applied?: boolean;
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
      duration_mode: "ai_per_track",
      album_writer_mode: "per_track_writer_loop",
      language: "en",
      song_model: "acestep-v15-xl-sft",
      audio_backend: "mps_torch",
      song_model_strategy: "single_model_album",
      quality_profile: "standard",
      style_profile: "auto",
      custom_tags: "",
      negative_tags: "",
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
      "video_prompt",
    ] as const) {
      if (k in payload) {
        // @ts-expect-error dynamic
        next[k] = payload[k];
      }
    }
    if (Array.isArray(payload.tracks)) {
      const hydratedTracks = normalizeAlbumTracks(payload.tracks, Number(next.track_duration ?? values.track_duration ?? 180));
      next.tracks = hydratedTracks;
      next.num_tracks = next.num_tracks ?? hydratedTracks.length;
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

  const setLoraSelection = (selection: LoraSelection) => {
    form.setValue("use_lora", selection.use_lora, { shouldValidate: true });
    form.setValue("lora_adapter_path", selection.lora_adapter_path, { shouldValidate: true });
    form.setValue("lora_adapter_name", selection.lora_adapter_name, { shouldValidate: true });
    form.setValue("use_lora_trigger", selection.use_lora_trigger, { shouldValidate: true });
    form.setValue("lora_trigger_tag", selection.lora_trigger_tag, { shouldValidate: true });
    form.setValue("lora_scale", selection.lora_scale, { shouldValidate: true });
    form.setValue("adapter_model_variant", selection.adapter_model_variant, { shouldValidate: true });
    form.setValue("adapter_song_model", selection.adapter_song_model, { shouldValidate: true });
    if (selection.use_lora && selection.adapter_song_model) {
      form.setValue("song_model", selection.adapter_song_model, { shouldValidate: true });
      form.setValue("song_model_strategy", "single_model_album", { shouldValidate: true });
    }
    draftState.saveNow({
      ...form.getValues(),
      ...selection,
      ...(selection.use_lora && selection.adapter_song_model
        ? { song_model: selection.adapter_song_model, song_model_strategy: "single_model_album" as const }
        : {}),
    });
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

  const albumCurrentPayload = React.useMemo(
    () => ({
      ...form.getValues(),
      use_mlx_dit: useMlxDitForAudioBackend(values.audio_backend),
      ...normalizeLoraSelection(values),
      planner_lm_provider: plannerProvider,
      ollama_model: plannerModel || undefined,
      planner_model: plannerModel || undefined,
      embedding_provider: embeddingProvider,
      embedding_lm_provider: embeddingProvider,
      embedding_model: embeddingModel || undefined,
      ace_step_text_encoder: "Qwen3-Embedding-0.6B",
      tracks: reviewTracks,
    }),
    [embeddingModel, embeddingProvider, form, plannerModel, plannerProvider, reviewTracks, values],
  );

  // ---- Async generate ----
  const startJob = useMutation({
    mutationFn: () => {
      const body = {
        concept: values.concept,
        album_title: values.album_title,
        artist_name: values.artist_name,
        num_tracks: values.num_tracks,
        track_duration: values.track_duration,
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
        video_prompt: values.video_prompt,
        ...normalizeLoraSelection(values),
        planner_lm_provider: plannerProvider,
        ollama_model: plannerModel || undefined,
        planner_model: plannerModel || undefined,
        embedding_provider: embeddingProvider,
        embedding_lm_provider: embeddingProvider,
        embedding_model: embeddingModel || undefined,
        ace_step_text_encoder: "Qwen3-Embedding-0.6B",
        tracks: reviewTracks,
      };
      return startAlbumPlanJob(body);
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
                            next[idx] = { ...next[idx], duration: Number(e.target.value) || values.track_duration };
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
                        <Input
                          value={t.key_scale ?? ""}
                          onChange={(e) => {
                            const next = normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration);
                            next[idx] = { ...next[idx], key_scale: e.target.value };
                            updatePlanTracks(next);
                          }}
                          placeholder="D minor"
                        />
                      </div>
                    </div>
                    <div className="space-y-1">
                      <Label className="text-xs">Time signature</Label>
                      <Input
                        value={t.time_signature ?? ""}
                        onChange={(e) => {
                          const next = normalizeAlbumTracks(values.tracks?.length ? values.tracks : plan?.tracks, values.track_duration);
                          next[idx] = { ...next[idx], time_signature: e.target.value };
                          updatePlanTracks(next);
                        }}
                        placeholder="4"
                      />
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
                      {Object.keys(trackQuality(t).rapBarCounts).length > 0 && (
                        <Badge variant="outline" className="text-[10px]">
                          rap bars {Object.values(trackQuality(t).rapBarCounts).map((count) => String(count)).join("/")}
                        </Badge>
                      )}
                      {(t.payload_gate_status || trackQuality(t).gateStatus) && (
                        <Badge variant={(t.payload_gate_status || trackQuality(t).gateStatus) === "pass" ? "default" : "destructive"} className="text-[10px]">
                          gate {t.payload_gate_status || trackQuality(t).gateStatus}
                        </Badge>
                      )}
                      {values.use_lora && (
                        <Badge variant="outline" className="text-[10px]">
                          LoRA {values.lora_adapter_name || t.lora_adapter_name || "actief"}
                          {values.lora_trigger_tag ? ` · ${values.lora_trigger_tag}` : ""}
                          {typeof values.lora_scale === "number" ? ` · ${Math.round(values.lora_scale * 100)}%` : ""}
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
                <Input {...form.register("language")} />
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
          <FieldGroup
            title="LoRA"
            description="Optioneel: deze PEFT LoRA wordt op elke albumtrack toegepast."
          >
            <LoraSelector value={values} onChange={setLoraSelection} />
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
              { key: "duration_mode", label: "Duurbeleid", format: (v) => v === "fixed" ? "Vaste duur" : "AI per track" },
              { key: "album_writer_mode", label: "AI writer", format: () => "Per-track loop" },
              { key: "track_duration", label: "Fallback duur", format: (v) => formatDuration(Number(v) || 0) },
              { key: "song_model", label: "Model" },
              { key: "audio_backend", label: "Backend", format: audioBackendLabel },
              { key: "planner_model", label: "AI planner", format: (v) => `${PROVIDER_LABEL[plannerProvider]} · ${String(v || "—")}` },
              { key: "embedding_model", label: "AI Memory", format: (v) => `${PROVIDER_LABEL[embeddingProvider]} · ${String(v || "—")}` },
              { key: "ace_step_text_encoder", label: "ACE-Step encoder" },
              { key: "lora_adapter_name", label: "LoRA" },
              { key: "lora_trigger_tag", label: "LoRA trigger" },
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
              {resultTracks.map((t, i) => (
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
                  {t.audio_url && (
                    <div className="space-y-2">
                      <WaveformPlayer
                        src={t.audio_url}
                        title={t.title}
                        artist={values.artist_name}
                      />
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => navigate("/wizard/video", {
                          state: {
                            audio_url: t.audio_url,
                            title: t.title,
                            artist_name: values.artist_name,
                            prompt: String(t.caption || values.concept || t.title || ""),
                            target_type: "song",
                            target_id: t.result_id || t.song_id || `${lastResult.album_id || albumFamilyId || "album"}:track:${i + 1}`,
                          },
                        })}
                        className="gap-2"
                      >
                        <Video className="size-3.5" />
                        Create video
                      </Button>
                    </div>
                  )}
                </motion.div>
              ))}
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
  return {
    wordCount,
    lineCount,
    sectionCount: Number(quality.section_count ?? (sectionCount || 0)),
    hookCount,
    rapBarCounts,
    gateStatus: String(track.payload_gate_status || quality.gate_status || ""),
    targetWords: Number(quality.target_words || 0),
    minWords: Number(quality.min_words || 0),
  };
}
