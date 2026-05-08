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
} from "@/lib/api";
import { DEFAULT_LORA_SCALE, normalizeLoraSelection, type LoraSelection } from "@/lib/lora";
import { mergeWizardDraft, usePromptMirror, useWizardDraft } from "@/hooks/useWizardDraft";
import { useWizardStore } from "@/store/wizard";
import { useSettingsStore } from "@/store/settings";
import { useJobsStore } from "@/store/jobs";
import { toast } from "@/components/ui/sonner";
import { cn, formatDuration } from "@/lib/utils";

const MODE = "album" as const;

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
  tags?: string;
  lyrics?: string;
  bpm?: number;
  key_scale?: string;
  role?: string;
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

export function AlbumWizard() {
  const navigate = useNavigate();
  const setResult = useWizardStore((s) => s.setResult);
  const lastResult = useWizardStore((s) => s.lastResult[MODE]);
  const warnings = useWizardStore((s) => s.warnings[MODE]) ?? [];
  const storePrompt = useWizardStore((s) => s.prompts[MODE]);
  const draft = useWizardStore((s) => s.drafts[MODE]);
  const plannerModel = useSettingsStore((s) => s.plannerModel);
  const albumDefaults = React.useMemo<AlbumFormValues>(
    () => ({
      concept: "",
      num_tracks: 7,
      track_duration: 180,
      language: "en",
      song_model: "acestep-v15-xl-sft",
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
      "language",
      "song_model",
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
    const merged = { ...form.getValues(), ...next };
    form.reset(merged);
    if (Array.isArray(payload.tracks)) {
      setPlan({
        concept: (payload.concept as string) ?? values.concept,
        album_title: payload.album_title as string,
        artist_name: payload.artist_name as string,
        num_tracks:
          typeof payload.num_tracks === "number" ? payload.num_tracks : undefined,
        language: payload.language as string,
        song_model: payload.song_model as string,
        tracks: payload.tracks as AlbumTrack[],
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
  };

  // ---- Async generate ----
  const startJob = useMutation({
    mutationFn: () => {
      const body = {
        concept: values.concept,
        album_title: values.album_title,
        artist_name: values.artist_name,
        num_tracks: values.num_tracks,
        track_duration: values.track_duration,
        language: values.language,
        song_model: values.song_model,
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
        planner_lm_provider: "ollama",
        ollama_model: plannerModel || undefined,
        planner_model: plannerModel || undefined,
        tracks: plan?.tracks,
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
          toast.error(`Polling-fout: ${(e as Error).message}`);
          updateJob(jobId, { status: "error", state: "error", error: (e as Error).message });
          setJobId(null);
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
                  <Label>Duur per track</Label>
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
              {plan?.tracks?.length ?? 0} tracks gepland
            </p>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                const next = [...(plan?.tracks ?? [])];
                next.push({
                  track_number: next.length + 1,
                  title: `Track ${next.length + 1}`,
                  duration: values.track_duration,
                });
                setPlan({ ...(plan ?? {}), tracks: next });
              }}
              className="gap-1.5"
            >
              <Plus className="size-3.5" /> Track toevoegen
            </Button>
          </div>
          <AnimatePresence initial={false}>
            {(plan?.tracks ?? []).map((t, idx) => (
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
                        const next = [...(plan?.tracks ?? [])];
                        next[idx] = { ...next[idx], title: e.target.value };
                        setPlan({ ...(plan ?? {}), tracks: next });
                      }}
                      placeholder="Tracktitel"
                    />
                    <Textarea
                      rows={2}
                      value={t.caption ?? ""}
                      onChange={(e) => {
                        const next = [...(plan?.tracks ?? [])];
                        next[idx] = { ...next[idx], caption: e.target.value };
                        setPlan({ ...(plan ?? {}), tracks: next });
                      }}
                      placeholder="Caption / vibe"
                    />
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
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={() => {
                      const next = (plan?.tracks ?? []).filter((_, i) => i !== idx);
                      setPlan({ ...(plan ?? {}), tracks: next });
                    }}
                    title="Verwijder track"
                  >
                    <Trash2 className="size-3.5" />
                  </Button>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          {(plan?.tracks?.length ?? 0) === 0 && (
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
            <div className="grid gap-3 sm:grid-cols-2">
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
              ...form.getValues(),
              ...normalizeLoraSelection(values),
              tracks: plan?.tracks,
            }}
            warnings={warnings}
            primaryFields={[
              { key: "album_title", label: "Album titel" },
              { key: "artist_name", label: "Artiest" },
              { key: "num_tracks", label: "Tracks" },
              { key: "track_duration", label: "Duur/track", format: (v) => formatDuration(Number(v) || 0) },
              { key: "song_model", label: "Model" },
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
