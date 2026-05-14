import * as React from "react";
import {
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Copy,
  ListMusic,
  Plus,
  RefreshCw,
  Trash2,
} from "lucide-react";

import { AudioBackendSelector } from "@/components/wizard/AudioBackendSelector";
import { AudioStyleSelector } from "@/components/wizard/AudioStyleSelector";
import { AceStepAdvancedSettings } from "@/components/wizard/AceStepAdvancedSettings";
import { GenerationAudioList } from "@/components/wizard/GenerationAudioList";
import { LoraSelector } from "@/components/wizard/LoraSelector";
import { TagInput } from "@/components/wizard/TagInput";
import { FieldGroup } from "@/components/wizard/WizardShell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "@/components/ui/sonner";
import { getSongBatchJob, startSongBatchJob, type SongBatchJob } from "@/lib/api";
import {
  ACE_STEP_ADVANCED_DEFAULTS,
  ACE_STEP_ADVANCED_PAYLOAD_FIELDS,
  ACE_STEP_KEY_SCALE_OPTIONS,
  ACE_STEP_TIME_SIGNATURE_OPTIONS,
  OFFICIAL_AUDIO_FORMAT_OPTIONS,
} from "@/lib/aceStepSettings";
import { ACE_STEP_LANGUAGE_OPTIONS } from "@/lib/languages";
import { normalizeLoraSelection, type LoraSelection } from "@/lib/lora";
import { useMlxDitForAudioBackend, type AudioBackend } from "@/lib/audioBackend";
import { simpleDefaults, type CustomFormValues } from "@/lib/schemas";
import { cn, formatDuration } from "@/lib/utils";
import { useJobsStore } from "@/store/jobs";

const SONG_MODELS = [
  ["acestep-v15-xl-sft", "ACE-Step v1.5 XL SFT"],
  ["acestep-v15-xl-base", "ACE-Step v1.5 XL Base"],
  ["acestep-v15-xl-turbo", "ACE-Step v1.5 XL Turbo"],
  ["acestep-v15-sft", "ACE-Step v1.5 SFT"],
  ["acestep-v15-base", "ACE-Step v1.5 Base"],
  ["acestep-v15-turbo", "ACE-Step v1.5 Turbo"],
  ["acestep-v15-turbo-shift1", "ACE-Step v1.5 Turbo (shift 1)"],
] as const;

const QUALITY_PROFILES = [
  ["draft", "Laag"],
  ["standard", "Middel"],
  ["chart_master", "Hoog"],
] as const;

const TASK_TYPES = [
  ["text2music", "Text to music"],
  ["cover", "Cover"],
  ["repaint", "Repaint"],
  ["extract", "Extract"],
  ["lego", "Lego"],
  ["complete", "Complete"],
] as const;

const TAG_SUGGESTIONS = [
  "rap",
  "hip hop",
  "rnb",
  "soul",
  "pop",
  "rock",
  "edm",
  "cinematic",
  "country",
  "deep bass",
  "hard drums",
  "clear vocal",
  "radio-ready mix",
  "analog warmth",
  "wide stereo",
  "tight low end",
];

const NEG_SUGGESTIONS = [
  "generic lyrics",
  "weak hook",
  "muddy mix",
  "off-key vocals",
  "clipping",
  "low quality",
  "mumbled vocals",
  "thin drums",
];

type SongDraft = CustomFormValues & { id: string };

type GlobalDefaults = Pick<
  SongDraft,
  | "song_model"
  | "audio_backend"
  | "quality_profile"
  | "style_profile"
  | "vocal_language"
  | "use_lora"
  | "lora_adapter_path"
  | "lora_adapter_name"
  | "use_lora_trigger"
  | "lora_trigger_tag"
  | "lora_scale"
  | "adapter_model_variant"
  | "adapter_song_model"
>;

function docsCorrectRenderDefaults(songModel: string) {
  if (songModel.includes("turbo")) {
    return { inference_steps: 8, shift: 3 };
  }
  return { inference_steps: 50, shift: 1 };
}

function uid() {
  return Math.random().toString(36).slice(2, 10);
}

function makeSong(seed: Partial<SongDraft> = {}, index = 0): SongDraft {
  const songModel = String(seed.song_model || "acestep-v15-xl-sft");
  const defaults = docsCorrectRenderDefaults(songModel);
  const advancedDefaults = ACE_STEP_ADVANCED_DEFAULTS as Partial<SongDraft>;
  const advancedOverrides = Object.fromEntries(
    ACE_STEP_ADVANCED_PAYLOAD_FIELDS.map((field) => [
      field,
      seed[field as keyof SongDraft] ?? advancedDefaults[field as keyof SongDraft],
    ]),
  ) as Partial<SongDraft>;
  return {
    ...simpleDefaults,
    task_type: "text2music",
    simple_description: "",
    title: seed.title || `Batch song ${index + 1}`,
    artist_name: seed.artist_name || "",
    caption:
      seed.caption ||
      "rap, hip hop, clear lead vocal, hard drums, deep bass, polished full mix",
    style_profile: seed.style_profile || "rap",
    tags: seed.tags || "rap, hip hop, hard drums, deep bass",
    negative_tags: seed.negative_tags || "generic lyrics, muddy mix, mumbled vocals",
    lyrics:
      seed.lyrics ||
      "[Verse - rap, rhythmic spoken flow]\nI write the first line with pressure in the pocket\nKick hits low while the city lights watch it\n\n[Chorus - rap hook]\nRun it one time, make the whole room lock in\nBassline heavy, every bar keeps knocking",
    instrumental: seed.instrumental ?? false,
    duration: Number(seed.duration ?? 180),
    bpm: seed.bpm ?? 92,
    key_scale: seed.key_scale || "D minor",
    time_signature: seed.time_signature || "4/4",
    vocal_language: seed.vocal_language || "en",
    song_model: songModel,
    audio_backend: seed.audio_backend || "mps_torch",
    quality_profile: seed.quality_profile || "chart_master",
    seed: seed.seed ?? -1,
    inference_steps: Number(seed.inference_steps ?? defaults.inference_steps),
    guidance_scale: Number(seed.guidance_scale ?? 7),
    shift: Number(seed.shift ?? defaults.shift),
    audio_format: seed.audio_format || "wav32",
    batch_size: Number(seed.batch_size ?? 1),
    ...advancedDefaults,
    ...advancedOverrides,
    auto_song_art: seed.auto_song_art ?? false,
    auto_album_art: seed.auto_album_art ?? false,
    auto_video_clip: seed.auto_video_clip ?? false,
    art_prompt: seed.art_prompt || "",
    video_prompt: seed.video_prompt || "",
    use_lora: seed.use_lora ?? false,
    lora_adapter_path: seed.lora_adapter_path || "",
    lora_adapter_name: seed.lora_adapter_name || "",
    use_lora_trigger: seed.use_lora_trigger ?? false,
    lora_trigger_tag: seed.lora_trigger_tag || "",
    lora_scale: Number(seed.lora_scale ?? simpleDefaults.lora_scale),
    adapter_model_variant: seed.adapter_model_variant || "",
    adapter_song_model: seed.adapter_song_model || "",
    id: seed.id || uid(),
  };
}

function buildPayload(song: SongDraft): Record<string, unknown> {
  const advanced: Record<string, unknown> = {};
  for (const key of ACE_STEP_ADVANCED_PAYLOAD_FIELDS) {
    const value = song[key as keyof SongDraft];
    if (value === undefined || value === "") continue;
    advanced[key] = value;
  }
  return {
    task_type: song.task_type,
    title: song.title,
    artist_name: song.artist_name,
    caption: song.caption,
    style_profile: song.style_profile,
    tags: song.tags,
    negative_tags: song.negative_tags,
    lyrics: song.instrumental ? "[Instrumental]" : song.lyrics,
    instrumental: song.instrumental,
    audio_duration: song.duration,
    duration: song.duration,
    bpm: song.bpm,
    key_scale: song.key_scale,
    time_signature: song.time_signature,
    vocal_language: song.vocal_language,
    song_model: song.song_model,
    audio_backend: song.audio_backend,
    use_mlx_dit: useMlxDitForAudioBackend(song.audio_backend),
    quality_profile: song.quality_profile,
    seed: song.seed,
    inference_steps: song.inference_steps,
    guidance_scale: song.guidance_scale,
    shift: song.shift,
    audio_format: song.audio_format,
    batch_size: song.batch_size,
    ...advanced,
    auto_song_art: song.auto_song_art,
    auto_album_art: false,
    auto_video_clip: song.auto_video_clip,
    art_prompt: song.art_prompt,
    video_prompt: song.video_prompt,
    wizard_mode: "batch",
    ...normalizeLoraSelection(song),
  };
}

function songFromJson(value: unknown, index: number): SongDraft {
  const record = value && typeof value === "object" && !Array.isArray(value)
    ? (value as Partial<SongDraft>)
    : {};
  return makeSong(record, index);
}

function batchJson(batchTitle: string, stopOnError: boolean, songs: SongDraft[]) {
  return JSON.stringify(
    {
      batch_title: batchTitle,
      stop_on_error: stopOnError,
      songs: songs.map(buildPayload),
    },
    null,
    2,
  );
}

function stateDone(state?: string) {
  return ["succeeded", "failed", "error", "stopped", "completed", "complete"].includes(
    String(state || "").toLowerCase(),
  );
}

export function BatchSongsWizard() {
  const addJob = useJobsStore((s) => s.addJob);
  const openJob = useJobsStore((s) => s.openJob);
  const [batchTitle, setBatchTitle] = React.useState("Batch Songs");
  const [stopOnError, setStopOnError] = React.useState(false);
  const [songs, setSongs] = React.useState<SongDraft[]>(() => [makeSong({}, 0)]);
  const [selectedId, setSelectedId] = React.useState(() => songs[0]?.id || "");
  const [jsonText, setJsonText] = React.useState(() => batchJson("Batch Songs", false, songs));
  const [jsonDirty, setJsonDirty] = React.useState(false);
  const [submitting, setSubmitting] = React.useState(false);
  const [activeJob, setActiveJob] = React.useState<SongBatchJob | null>(null);
  const [error, setError] = React.useState("");

  React.useEffect(() => {
    if (!jsonDirty) {
      setJsonText(batchJson(batchTitle, stopOnError, songs));
    }
  }, [batchTitle, stopOnError, songs, jsonDirty]);

  React.useEffect(() => {
    if (!activeJob?.id || stateDone(activeJob.state)) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const resp = await getSongBatchJob(activeJob.id);
        if (cancelled) return;
        if (resp.job) {
          setActiveJob(resp.job);
          addJob({
            id: resp.job.id,
            kind: "song-batch",
            label: resp.job.batch_title || "Batch Songs",
            progress: resp.job.progress ?? 0,
            status: resp.job.status || resp.job.stage || "running",
            state: resp.job.state || "running",
            stage: resp.job.stage,
            kindLabel: "Song batch",
            detailsPath: `/api/song-batches/jobs/${encodeURIComponent(resp.job.id)}`,
            logPath: `/api/song-batches/jobs/${encodeURIComponent(resp.job.id)}/log`,
            metadata: resp.job as unknown as Record<string, unknown>,
            error: resp.job.error || resp.job.errors?.join("\n") || "",
            startedAt: resp.job.started_at ? new Date(resp.job.started_at).getTime() : Date.now(),
            updatedAt: resp.job.updated_at,
          });
        }
      } catch (pollError) {
        if (!cancelled) setError(pollError instanceof Error ? pollError.message : "Batch poll failed");
      }
    };
    const interval = window.setInterval(tick, 2500);
    void tick();
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [activeJob?.id, activeJob?.state, addJob]);

  const selectedSong = songs.find((song) => song.id === selectedId) || songs[0];

  const updateSong = (id: string, patch: Partial<SongDraft>) => {
    setSongs((current) =>
      current.map((song) => {
        if (song.id !== id) return song;
        const next = { ...song, ...patch };
        if (patch.song_model && patch.song_model !== song.song_model) {
          const defaults = docsCorrectRenderDefaults(String(patch.song_model));
          next.inference_steps = defaults.inference_steps;
          next.shift = defaults.shift;
        }
        return next;
      }),
    );
  };

  const setLoraSelection = (id: string, selection: LoraSelection) => {
    const patch: Partial<SongDraft> = {
      use_lora: selection.use_lora,
      lora_adapter_path: selection.lora_adapter_path,
      lora_adapter_name: selection.lora_adapter_name,
      use_lora_trigger: selection.use_lora_trigger,
      lora_trigger_tag: selection.lora_trigger_tag,
      lora_scale: selection.lora_scale,
      adapter_model_variant: selection.adapter_model_variant,
      adapter_song_model: selection.adapter_song_model,
    };
    if (selection.use_lora && selection.adapter_song_model) {
      const defaults = docsCorrectRenderDefaults(selection.adapter_song_model);
      patch.song_model = selection.adapter_song_model;
      patch.inference_steps = defaults.inference_steps;
      patch.shift = defaults.shift;
    }
    updateSong(id, patch);
  };

  const addSong = () => {
    const next = makeSong(songs[songs.length - 1] || {}, songs.length);
    next.id = uid();
    next.title = `Batch song ${songs.length + 1}`;
    setSongs((current) => [...current, next]);
    setSelectedId(next.id);
  };

  const duplicateSong = (song: SongDraft) => {
    const next = makeSong({ ...song, id: uid(), title: `${song.title || "Song"} copy` }, songs.length);
    setSongs((current) => [...current, next]);
    setSelectedId(next.id);
  };

  const removeSong = (id: string) => {
    setSongs((current) => {
      if (current.length <= 1) return current;
      const next = current.filter((song) => song.id !== id);
      if (selectedId === id) setSelectedId(next[0]?.id || "");
      return next;
    });
  };

  const moveSong = (id: string, direction: -1 | 1) => {
    setSongs((current) => {
      const index = current.findIndex((song) => song.id === id);
      const target = index + direction;
      if (index < 0 || target < 0 || target >= current.length) return current;
      const next = [...current];
      const [item] = next.splice(index, 1);
      next.splice(target, 0, item);
      return next;
    });
  };

  const applyGlobalDefaults = () => {
    if (!selectedSong) return;
    const source: GlobalDefaults = {
      song_model: selectedSong.song_model,
      audio_backend: selectedSong.audio_backend,
      quality_profile: selectedSong.quality_profile,
      style_profile: selectedSong.style_profile,
      vocal_language: selectedSong.vocal_language,
      use_lora: selectedSong.use_lora,
      lora_adapter_path: selectedSong.lora_adapter_path,
      lora_adapter_name: selectedSong.lora_adapter_name,
      use_lora_trigger: selectedSong.use_lora_trigger,
      lora_trigger_tag: selectedSong.lora_trigger_tag,
      lora_scale: selectedSong.lora_scale,
      adapter_model_variant: selectedSong.adapter_model_variant,
      adapter_song_model: selectedSong.adapter_song_model,
    };
    const defaults = docsCorrectRenderDefaults(source.song_model);
    setSongs((current) =>
      current.map((song) => ({
        ...song,
        ...source,
        inference_steps: defaults.inference_steps,
        shift: defaults.shift,
      })),
    );
    toast.success("Defaults op alle songs toegepast.");
  };

  const applyJson = () => {
    try {
      const parsed = JSON.parse(jsonText);
      const body = Array.isArray(parsed) ? { songs: parsed } : parsed;
      if (!body || typeof body !== "object" || !Array.isArray((body as { songs?: unknown }).songs)) {
        throw new Error("JSON moet { songs: [...] } bevatten.");
      }
      const rawSongs = (body as { songs: unknown[] }).songs;
      if (rawSongs.length === 0) throw new Error("songs[] mag niet leeg zijn.");
      const nextSongs = rawSongs.map(songFromJson);
      setBatchTitle(String((body as { batch_title?: unknown; title?: unknown }).batch_title || (body as { title?: unknown }).title || "Batch Songs"));
      setStopOnError(Boolean((body as { stop_on_error?: unknown }).stop_on_error));
      setSongs(nextSongs);
      setSelectedId(nextSongs[0]?.id || "");
      setJsonDirty(false);
      toast.success("Batch JSON toegepast.");
    } catch (jsonError) {
      const first = makeSong({ title: "Pasted song", caption: jsonText, lyrics: jsonText }, 0);
      setSongs([first]);
      setSelectedId(first.id);
      setJsonDirty(false);
      toast.error(jsonError instanceof Error ? jsonError.message : "JSON kon niet gelezen worden");
    }
  };

  const startBatch = async () => {
    setSubmitting(true);
    setError("");
    try {
      const body = {
        batch_title: batchTitle,
        stop_on_error: stopOnError,
        songs: songs.map(buildPayload),
      };
      const resp = await startSongBatchJob(body);
      if (!resp.success || !resp.job_id) {
        throw new Error(resp.error || "Batch starten mislukt");
      }
      const job = resp.job || ({ id: resp.job_id, batch_title: batchTitle, state: "queued" } as SongBatchJob);
      setActiveJob(job);
      addJob({
        id: resp.job_id,
        kind: "song-batch",
        label: batchTitle || "Batch Songs",
        progress: job.progress ?? 0,
        status: job.status || "queued",
        state: job.state || "queued",
        stage: job.stage,
        kindLabel: "Song batch",
        detailsPath: `/api/song-batches/jobs/${encodeURIComponent(resp.job_id)}`,
        logPath: `/api/song-batches/jobs/${encodeURIComponent(resp.job_id)}/log`,
        metadata: job as unknown as Record<string, unknown>,
        startedAt: Date.now(),
      });
      openJob(resp.job_id);
      toast.success("Batch Songs queue gestart.");
    } catch (submitError) {
      const message = submitError instanceof Error ? submitError.message : "Batch starten mislukt";
      setError(message);
      toast.error(message);
    } finally {
      setSubmitting(false);
    }
  };

  const jobSongs = activeJob?.songs || [];
  const activeProgress = activeJob?.progress ?? 0;

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="border-b border-border/60 px-6 py-5 sm:px-10">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div className="space-y-1">
            <p className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
              Wizard
            </p>
            <h1 className="font-display text-2xl font-semibold sm:text-3xl">Batch Songs</h1>
            <p className="max-w-2xl text-sm text-muted-foreground">
              Stel meerdere complete songs in en laat de backend ze één voor één renderen. Klaar audio verschijnt direct, terwijl de rest doorloopt.
            </p>
          </div>
          <Button onClick={startBatch} size="lg" disabled={submitting || songs.length === 0}>
            {submitting ? <RefreshCw className="size-4 animate-spin" /> : <ListMusic className="size-4" />}
            Genereer batch
          </Button>
        </div>
      </header>

      <main className="min-h-0 flex-1 overflow-y-auto px-6 py-8 sm:px-10">
        <div className="mx-auto grid w-full max-w-7xl gap-6 xl:grid-cols-[300px_minmax(0,1fr)_360px]">
          <aside className="space-y-4">
            <FieldGroup title="Queue">
              <div className="space-y-3">
                <div className="space-y-1.5">
                  <Label>Batch titel</Label>
                  <Input value={batchTitle} onChange={(event) => setBatchTitle(event.target.value)} />
                </div>
                <div className="flex items-center justify-between gap-3 rounded-md border bg-background/35 p-3">
                  <Label>Stop bij fout</Label>
                  <Switch checked={stopOnError} onCheckedChange={setStopOnError} />
                </div>
                <Button type="button" className="w-full" onClick={addSong}>
                  <Plus className="size-4" />
                  Song toevoegen
                </Button>
              </div>
            </FieldGroup>

            <div className="space-y-2">
              {songs.map((song, index) => (
                <button
                  key={song.id}
                  type="button"
                  onClick={() => setSelectedId(song.id)}
                  className={cn(
                    "w-full rounded-md border p-3 text-left transition-colors",
                    selectedId === song.id
                      ? "border-primary/50 bg-primary/10"
                      : "bg-card/45 hover:border-primary/30",
                  )}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs text-muted-foreground">#{index + 1}</span>
                    <Badge variant="outline">{formatDuration(Number(song.duration || 0))}</Badge>
                  </div>
                  <p className="mt-1 truncate text-sm font-medium">{song.title || `Song ${index + 1}`}</p>
                  <p className="truncate text-xs text-muted-foreground">
                    {song.style_profile || "auto"} · {song.song_model}
                  </p>
                </button>
              ))}
            </div>
          </aside>

          <section className="min-w-0 space-y-5">
            {selectedSong && (
              <>
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <p className="text-xs uppercase tracking-wider text-muted-foreground">
                      Song {songs.findIndex((song) => song.id === selectedSong.id) + 1} / {songs.length}
                    </p>
                    <h2 className="font-display text-xl font-semibold">
                      {selectedSong.title || "Untitled song"}
                    </h2>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Button variant="outline" size="sm" onClick={() => moveSong(selectedSong.id, -1)}>
                      <ChevronUp className="size-4" />
                    </Button>
                    <Button variant="outline" size="sm" onClick={() => moveSong(selectedSong.id, 1)}>
                      <ChevronDown className="size-4" />
                    </Button>
                    <Button variant="outline" size="sm" onClick={() => duplicateSong(selectedSong)}>
                      <Copy className="size-4" />
                      Dupliceer
                    </Button>
                    <Button variant="destructive" size="sm" onClick={() => removeSong(selectedSong.id)} disabled={songs.length <= 1}>
                      <Trash2 className="size-4" />
                    </Button>
                  </div>
                </div>

                <FieldGroup title="Globale defaults" description="Gebruik de huidige song als template voor model, backend, stijl, taal, kwaliteit en LoRA.">
                  <Button type="button" variant="secondary" onClick={applyGlobalDefaults}>
                    Pas huidige instellingen toe op alle songs
                  </Button>
                </FieldGroup>

                <FieldGroup title="Identiteit">
                  <div className="grid gap-3 md:grid-cols-3">
                    <div className="space-y-1.5">
                      <Label>Titel</Label>
                      <Input value={selectedSong.title || ""} onChange={(event) => updateSong(selectedSong.id, { title: event.target.value })} />
                    </div>
                    <div className="space-y-1.5">
                      <Label>Artiest</Label>
                      <Input value={selectedSong.artist_name || ""} onChange={(event) => updateSong(selectedSong.id, { artist_name: event.target.value })} />
                    </div>
                    <div className="space-y-1.5">
                      <Label>Task type</Label>
                      <Select value={selectedSong.task_type} onValueChange={(value) => updateSong(selectedSong.id, { task_type: value as SongDraft["task_type"] })}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          {TASK_TYPES.map(([value, label]) => (
                            <SelectItem key={value} value={value}>{label}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </FieldGroup>

                <FieldGroup title="Genre, caption & tags">
                  <AudioStyleSelector value={selectedSong.style_profile} onChange={(value) => updateSong(selectedSong.id, { style_profile: value })} />
                  <div className="space-y-1.5">
                    <Label>Caption</Label>
                    <Textarea rows={4} value={selectedSong.caption || ""} onChange={(event) => updateSong(selectedSong.id, { caption: event.target.value })} />
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-1.5">
                      <Label>Tags</Label>
                      <TagInput value={selectedSong.tags} onChange={(value) => updateSong(selectedSong.id, { tags: value })} suggestions={TAG_SUGGESTIONS} />
                    </div>
                    <div className="space-y-1.5">
                      <Label>Negative tags</Label>
                      <TagInput value={selectedSong.negative_tags} onChange={(value) => updateSong(selectedSong.id, { negative_tags: value })} suggestions={NEG_SUGGESTIONS} variant="negative" />
                    </div>
                  </div>
                </FieldGroup>

                <FieldGroup title="Lyrics">
                  <div className="flex items-center justify-between gap-3 rounded-md border bg-background/35 p-3">
                    <Label>Instrumental</Label>
                    <Switch checked={selectedSong.instrumental} onCheckedChange={(checked) => updateSong(selectedSong.id, { instrumental: checked })} />
                  </div>
                  <Textarea
                    rows={14}
                    disabled={selectedSong.instrumental}
                    value={selectedSong.instrumental ? "[Instrumental]" : selectedSong.lyrics || ""}
                    onChange={(event) => updateSong(selectedSong.id, { lyrics: event.target.value })}
                    className="font-mono text-xs leading-relaxed"
                  />
                </FieldGroup>

                <FieldGroup title="Timing & metadata">
                  <div className="grid gap-3 md:grid-cols-4">
                    <div className="space-y-1.5">
                      <Label>Duur</Label>
                      <Input type="number" min={20} max={600} value={selectedSong.duration} onChange={(event) => updateSong(selectedSong.id, { duration: Number(event.target.value) })} />
                    </div>
                    <div className="space-y-1.5">
                      <Label>BPM</Label>
                      <Input type="number" value={selectedSong.bpm ?? ""} onChange={(event) => updateSong(selectedSong.id, { bpm: event.target.value ? Number(event.target.value) : undefined })} />
                    </div>
                    <div className="space-y-1.5">
                      <Label>Key</Label>
                      <Select
                        value={selectedSong.key_scale || "auto"}
                        onValueChange={(value) => updateSong(selectedSong.id, { key_scale: value === "auto" ? undefined : value })}
                      >
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          {ACE_STEP_KEY_SCALE_OPTIONS.map((value) => (
                            <SelectItem key={value} value={value}>{value === "auto" ? "Auto" : value}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-1.5">
                      <Label>Time signature</Label>
                      <Select
                        value={selectedSong.time_signature || "auto"}
                        onValueChange={(value) => updateSong(selectedSong.id, { time_signature: value === "auto" ? undefined : value })}
                      >
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          {ACE_STEP_TIME_SIGNATURE_OPTIONS.map(([value, label]) => (
                            <SelectItem key={value || "auto"} value={value || "auto"}>{label}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-1.5">
                      <Label>Taal</Label>
                      <Select value={selectedSong.vocal_language} onValueChange={(value) => updateSong(selectedSong.id, { vocal_language: value })}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          {ACE_STEP_LANGUAGE_OPTIONS.map(([code, label]) => (
                            <SelectItem key={code} value={code}>{label}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </FieldGroup>

                <FieldGroup title="ACE-Step render">
                  <div className="grid gap-3 md:grid-cols-3">
                    <div className="space-y-1.5">
                      <Label>Model</Label>
                      <Select value={selectedSong.song_model} onValueChange={(value) => updateSong(selectedSong.id, { song_model: value })}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          {SONG_MODELS.map(([value, label]) => (
                            <SelectItem key={value} value={value}>{label}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <AudioBackendSelector value={selectedSong.audio_backend} onChange={(value: AudioBackend) => updateSong(selectedSong.id, { audio_backend: value })} />
                    <div className="space-y-1.5">
                      <Label>Kwaliteit</Label>
                      <Select value={selectedSong.quality_profile} onValueChange={(value) => updateSong(selectedSong.id, { quality_profile: value as SongDraft["quality_profile"] })}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          {QUALITY_PROFILES.map(([value, label]) => (
                            <SelectItem key={value} value={value}>{label}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  <div className="grid gap-3 md:grid-cols-3">
                    <div className="space-y-1.5">
                      <Label>Steps</Label>
                      <Input type="number" min={4} max={100} value={selectedSong.inference_steps} onChange={(event) => updateSong(selectedSong.id, { inference_steps: Number(event.target.value) })} />
                    </div>
                    <div className="space-y-1.5">
                      <Label>Guidance</Label>
                      <Input type="number" min={1} max={15} step="0.1" value={selectedSong.guidance_scale} onChange={(event) => updateSong(selectedSong.id, { guidance_scale: Number(event.target.value) })} />
                    </div>
                    <div className="space-y-1.5">
                      <Label>Shift</Label>
                      <Input type="number" min={0} max={10} step="0.1" value={selectedSong.shift} onChange={(event) => updateSong(selectedSong.id, { shift: Number(event.target.value) })} />
                    </div>
                    <div className="space-y-1.5">
                      <Label>Audio format</Label>
                      <Select value={selectedSong.audio_format} onValueChange={(value) => updateSong(selectedSong.id, { audio_format: value as SongDraft["audio_format"] })}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          {OFFICIAL_AUDIO_FORMAT_OPTIONS.map(([value, label]) => (
                            <SelectItem key={value} value={value}>{label}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-1.5">
                      <Label>Seed</Label>
                      <Input type="number" value={selectedSong.seed ?? -1} onChange={(event) => updateSong(selectedSong.id, { seed: Number(event.target.value) })} />
                    </div>
                    <div className="space-y-1.5">
                      <Label>Takes</Label>
                      <div className="flex items-center gap-3">
                        <Slider
                          value={[selectedSong.batch_size]}
                          min={1}
                          max={8}
                          step={1}
                          onValueChange={(value) => updateSong(selectedSong.id, { batch_size: value[0] ?? 1 })}
                          className="flex-1"
                        />
                        <span className="w-8 text-right font-mono text-xs">{selectedSong.batch_size}</span>
                      </div>
                    </div>
                  </div>
                </FieldGroup>

                <FieldGroup
                  title="Official ACE-Step controls"
                  description="Alle extra ACE-Step velden voor DCW, CFG, output, retake, source/repaint, stems en runtime."
                >
                  <AceStepAdvancedSettings
                    values={selectedSong}
                    onChange={(key, value) =>
                      updateSong(selectedSong.id, { [key]: value } as Partial<SongDraft>)
                    }
                  />
                </FieldGroup>

                <FieldGroup title="LoRA">
                  <LoraSelector value={selectedSong} onChange={(selection) => setLoraSelection(selectedSong.id, selection)} />
                </FieldGroup>

                <FieldGroup title="Automation">
                  <div className="grid gap-3 md:grid-cols-2">
                    <label className="flex items-center justify-between gap-3 rounded-md border bg-background/35 p-3 text-sm">
                      <span>Song art</span>
                      <Switch
                        checked={selectedSong.auto_song_art}
                        onCheckedChange={(checked) => updateSong(selectedSong.id, { auto_song_art: checked })}
                      />
                    </label>
                    <label className="flex items-center justify-between gap-3 rounded-md border bg-background/35 p-3 text-sm">
                      <span>Video clip</span>
                      <Switch
                        checked={selectedSong.auto_video_clip}
                        onCheckedChange={(checked) => updateSong(selectedSong.id, { auto_video_clip: checked })}
                      />
                    </label>
                  </div>
                  {selectedSong.auto_song_art && (
                    <div className="space-y-1.5">
                      <Label>Art prompt override</Label>
                      <Textarea
                        rows={2}
                        value={selectedSong.art_prompt || ""}
                        onChange={(event) => updateSong(selectedSong.id, { art_prompt: event.target.value })}
                      />
                    </div>
                  )}
                  {selectedSong.auto_video_clip && (
                    <div className="space-y-1.5">
                      <Label>Video prompt override</Label>
                      <Textarea
                        rows={2}
                        value={selectedSong.video_prompt || ""}
                        onChange={(event) => updateSong(selectedSong.id, { video_prompt: event.target.value })}
                      />
                    </div>
                  )}
                </FieldGroup>
              </>
            )}
          </section>

          <aside className="space-y-5">
            <FieldGroup title="Batch JSON" description="Altijd bewerkbaar: plak hier je eigen queue of pas de gegenereerde JSON handmatig aan.">
              <div className="space-y-2">
                <Textarea
                  value={jsonText}
                  onChange={(event) => {
                    setJsonText(event.target.value);
                    setJsonDirty(true);
                  }}
                  rows={18}
                  className="font-mono text-[11px] leading-relaxed"
                />
                <div className="flex flex-wrap gap-2">
                  <Button type="button" variant="secondary" onClick={applyJson}>
                    JSON toepassen
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => {
                      setJsonText(batchJson(batchTitle, stopOnError, songs));
                      setJsonDirty(false);
                    }}
                  >
                    Sync uit formulier
                  </Button>
                </div>
              </div>
            </FieldGroup>

            {(error || activeJob) && (
              <FieldGroup title="Batch voortgang">
                {error && (
                  <p className="mb-3 flex gap-2 rounded-md bg-destructive/10 p-3 text-xs text-destructive">
                    <AlertTriangle className="size-4 shrink-0" />
                    {error}
                  </p>
                )}
                {activeJob && (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between gap-3">
                      <Badge variant={activeJob.state === "succeeded" ? "default" : activeJob.state === "failed" ? "destructive" : "secondary"}>
                        {activeJob.status || activeJob.state}
                      </Badge>
                      <span className="font-mono text-xs">{Math.round(activeProgress)}%</span>
                    </div>
                    <Progress value={activeProgress} className="h-1.5" />
                    <div className="grid grid-cols-3 gap-2 text-center text-xs">
                      <div className="rounded-md border bg-background/35 p-2">
                        <p className="font-mono">{activeJob.completed_songs ?? 0}</p>
                        <p className="text-muted-foreground">klaar</p>
                      </div>
                      <div className="rounded-md border bg-background/35 p-2">
                        <p className="font-mono">{activeJob.failed_songs ?? 0}</p>
                        <p className="text-muted-foreground">fout</p>
                      </div>
                      <div className="rounded-md border bg-background/35 p-2">
                        <p className="font-mono">{activeJob.remaining_songs ?? 0}</p>
                        <p className="text-muted-foreground">over</p>
                      </div>
                    </div>
                    <div className="space-y-3">
                      {jobSongs.map((song) => (
                        <div key={`${song.index}-${song.title}`} className="rounded-md border bg-background/35 p-3">
                          <div className="flex items-center justify-between gap-2">
                            <p className="truncate text-sm font-medium">{song.track_number}. {song.title}</p>
                            <Badge variant={song.state === "succeeded" ? "default" : song.state === "failed" ? "destructive" : "outline"}>
                              {song.status || song.state}
                            </Badge>
                          </div>
                          {song.progress !== undefined && <Progress value={song.progress} className="mt-2 h-1" />}
                          {song.error && <p className="mt-2 text-xs text-destructive">{song.error}</p>}
                          {song.result && (
                            <GenerationAudioList
                              result={song.result}
                              title={song.title}
                              artist={String(song.payload_summary?.artist_name || "")}
                              className="mt-3 space-y-2"
                            />
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </FieldGroup>
            )}
          </aside>
        </div>
      </main>
    </div>
  );
}
