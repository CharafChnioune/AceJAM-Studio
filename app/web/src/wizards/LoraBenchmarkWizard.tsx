import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import {
  BarChart3,
  CheckCircle2,
  Filter,
  Play,
  RefreshCw,
  Search,
  Star,
} from "lucide-react";

import { GenerationAudioList } from "@/components/wizard/GenerationAudioList";
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
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "@/components/ui/sonner";
import { api, getLoraAdapters, getLoraBenchmarkJob, rateLoraBenchmarkResult, startLoraBenchmarkJob, type LoraAdapter, type LoraBenchmarkJob, type LoraBenchmarkResult } from "@/lib/api";
import {
  ACE_STEP_KEY_SCALE_OPTIONS,
  ACE_STEP_TIME_SIGNATURE_OPTIONS,
  OFFICIAL_AUDIO_FORMAT_OPTIONS,
} from "@/lib/aceStepSettings";
import { useMlxDitForAudioBackend, type AudioBackend } from "@/lib/audioBackend";
import { ACE_STEP_LANGUAGE_OPTIONS } from "@/lib/languages";
import { loraAdapterLabel, loraTriggerOptions } from "@/lib/lora";
import { cn } from "@/lib/utils";
import { useJobsStore } from "@/store/jobs";

interface EpochAuditionGenre {
  key: string;
  label: string;
  caption_tags: string;
  lyrics_section_tags?: Record<string, string>;
  lyrics: string;
  test_lyrics?: string;
  bpm?: number | null;
  keyscale?: string;
  timesignature?: string;
  default?: boolean;
}

const FALLBACK_GENRES: EpochAuditionGenre[] = [
  {
    key: "rap",
    label: "Rap / Hip-hop",
    caption_tags: "rap, hip hop, rhythmic spoken-word vocal, clear rap flow, hard drums, deep bass, polished full mix",
    lyrics_section_tags: { verse: "rap, rhythmic spoken flow", chorus: "rap hook" },
    lyrics:
      "[Verse - rap, rhythmic spoken flow]\nI count the pressure by the kick when the room goes still\nEvery line got weight, every pocket got steel\nLow end breathing like a motor in the curbside heat\nOne take, same words, let the LoRA speak\n\n[Chorus - rap hook]\nRun the test loud, let the best one show\nSame drums, same bars, hear the trained voice grow",
    bpm: 92,
    keyscale: "D minor",
    timesignature: "4/4",
  },
  {
    key: "pop",
    label: "Pop",
    caption_tags: "modern pop, bright hook, clean lead vocal, punchy drums, radio-ready mix",
    lyrics:
      "[Verse - clean pop vocal]\nThe room lights up when the rhythm starts moving\nEvery little doubt gets lost in the groove\n\n[Chorus - bright pop hook]\nTurn it up now, let the best take shine\nOne melody, one moment, one clear line",
    bpm: 116,
    keyscale: "C major",
    timesignature: "4/4",
  },
  {
    key: "rnb",
    label: "Soul / R&B",
    caption_tags: "smooth rnb groove, warm keys, intimate lead vocal, soft harmonies, deep pocket",
    lyrics:
      "[Verse - smooth rnb vocal]\nLate night keys lean soft against the rain\nI hear your name in the space between the bass\n\n[Chorus - soulful rnb hook]\nHold close, move slow, let the feeling stay\nOne clear voice carries everything we say",
    bpm: 86,
    keyscale: "A minor",
    timesignature: "4/4",
  },
];

const SONG_MODELS = [
  ["acestep-v15-xl-sft", "XL SFT"],
  ["acestep-v15-xl-base", "XL Base"],
  ["acestep-v15-xl-turbo", "XL Turbo"],
  ["acestep-v15-sft", "SFT"],
  ["acestep-v15-base", "Base"],
  ["acestep-v15-turbo", "Turbo"],
] as const;

const QUALITY_PROFILES = [
  ["draft", "Laag"],
  ["standard", "Middel"],
  ["chart_master", "Hoog"],
] as const;

function docsCorrectDefaults(songModel: string) {
  return songModel.includes("turbo") ? { inference_steps: 8, shift: 3 } : { inference_steps: 50, shift: 1 };
}

function text(value: unknown, fallback = ""): string {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function number(value: unknown, fallback = 0): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function adapterQuality(adapter: LoraAdapter): string {
  return text(adapter.metadata?.quality_status || (adapter as unknown as Record<string, unknown>).quality_status || (adapter.generation_loadable ? "loadable" : "unknown"));
}

function adapterEpoch(adapter: LoraAdapter): number | null {
  const direct = number(adapter.metadata?.epoch ?? adapter.metadata?.best_loss_epoch, Number.NaN);
  if (Number.isFinite(direct)) return direct;
  const match = `${adapter.name} ${adapter.display_name || ""} ${adapter.path}`.match(/epoch[_\-\s]*(\d+)/i);
  return match ? Number(match[1]) : null;
}

function adapterLoss(adapter: LoraAdapter): number | null {
  const direct = number(adapter.metadata?.loss ?? adapter.metadata?.last_loss ?? adapter.metadata?.best_loss, Number.NaN);
  if (Number.isFinite(direct)) return direct;
  const match = `${adapter.name} ${adapter.display_name || ""} ${adapter.path}`.match(/loss[_\-\s]*(\d+(?:\.\d+)?)/i);
  return match ? Number(match[1]) : null;
}

function parseScales(value: string): number[] {
  const scales = value
    .split(/[,\s]+/)
    .map((item) => Number(item.trim()))
    .filter((item) => Number.isFinite(item))
    .map((item) => Math.max(0, Math.min(1, item)));
  return Array.from(new Set(scales.map((item) => item.toFixed(4)))).map(Number).slice(0, 8);
}

function stateDone(state?: string) {
  return ["succeeded", "failed", "error", "stopped", "completed", "complete"].includes(String(state || "").toLowerCase());
}

function resultScore(result: LoraBenchmarkResult): number {
  return Math.max(0, Math.min(100, number(result.score, 0)));
}

function resultLabel(result: LoraBenchmarkResult) {
  const epoch = result.adapter_epoch ? `epoch ${result.adapter_epoch}` : "";
  const scale = typeof result.lora_scale === "number" ? `${Math.round(result.lora_scale * 100)}%` : "";
  return [result.adapter_name || "No LoRA", epoch, scale].filter(Boolean).join(" · ");
}

function resultTranscript(result: LoraBenchmarkResult): string {
  const preview = result.transcript_preview;
  return Array.isArray(preview) ? preview.map((item) => text(item)).join(" / ") : text(preview);
}

function BarGraph({ results, bestId }: { results: LoraBenchmarkResult[]; bestId?: string }) {
  const rows = results.filter((item) => item.attempt_id);
  if (!rows.length) {
    return <p className="text-sm text-muted-foreground">Nog geen benchmarkdata.</p>;
  }
  return (
    <div className="space-y-2">
      {rows.map((row) => {
        const score = resultScore(row);
        const best = row.attempt_id === bestId;
        return (
          <div key={row.attempt_id} className="grid gap-2 sm:grid-cols-[minmax(120px,240px)_1fr_48px] sm:items-center">
            <p className="truncate text-xs text-muted-foreground">{resultLabel(row)}</p>
            <div className="h-3 overflow-hidden rounded-full bg-muted">
              <div
                className={cn("h-full", best ? "bg-emerald-500" : "bg-primary")}
                style={{ width: `${score}%` }}
              />
            </div>
            <p className="text-right font-mono text-xs">{score.toFixed(1)}</p>
          </div>
        );
      })}
    </div>
  );
}

function LossScoreGraph({ results }: { results: LoraBenchmarkResult[] }) {
  const rows = results.filter((item) => typeof item.adapter_epoch === "number" && typeof item.adapter_loss === "number");
  if (!rows.length) return <p className="text-sm text-muted-foreground">Loss/epoch grafiek verschijnt zodra adapters die metadata hebben klaar zijn.</p>;
  const maxLoss = Math.max(...rows.map((item) => number(item.adapter_loss, 0)), 0.001);
  return (
    <div className="grid gap-3 md:grid-cols-2">
      <div className="space-y-2">
        <p className="text-xs font-medium text-muted-foreground">Score per epoch</p>
        {rows.map((row) => (
          <div key={`score-${row.attempt_id}`} className="grid grid-cols-[72px_1fr_42px] items-center gap-2">
            <span className="font-mono text-xs">e{row.adapter_epoch}</span>
            <div className="h-2 overflow-hidden rounded-full bg-muted">
              <div className="h-full bg-primary" style={{ width: `${resultScore(row)}%` }} />
            </div>
            <span className="text-right font-mono text-xs">{resultScore(row).toFixed(0)}</span>
          </div>
        ))}
      </div>
      <div className="space-y-2">
        <p className="text-xs font-medium text-muted-foreground">Loss per epoch</p>
        {rows.map((row) => {
          const loss = number(row.adapter_loss, 0);
          const width = Math.max(2, Math.min(100, (loss / maxLoss) * 100));
          return (
            <div key={`loss-${row.attempt_id}`} className="grid grid-cols-[72px_1fr_54px] items-center gap-2">
              <span className="font-mono text-xs">e{row.adapter_epoch}</span>
              <div className="h-2 overflow-hidden rounded-full bg-muted">
                <div className="h-full bg-amber-500" style={{ width: `${width}%` }} />
              </div>
              <span className="text-right font-mono text-xs">{loss.toFixed(3)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function LoraBenchmarkWizard() {
  const addJob = useJobsStore((s) => s.addJob);
  const openJob = useJobsStore((s) => s.openJob);
  const adaptersQuery = useQuery({
    queryKey: ["lora", "benchmark-adapters"],
    queryFn: getLoraAdapters,
    staleTime: 15_000,
  });
  const genresQuery = useQuery({
    queryKey: ["lora-epoch-audition-genres"],
    queryFn: () => api.get<{ success: boolean; genres?: EpochAuditionGenre[] }>("/api/lora/epoch-audition/genres"),
    staleTime: 5 * 60 * 1000,
  });

  const genres = React.useMemo(() => {
    const fromApi = genresQuery.data?.genres?.filter((item) => item.key) ?? [];
    return fromApi.length ? fromApi : FALLBACK_GENRES;
  }, [genresQuery.data?.genres]);
  const [benchmarkTitle, setBenchmarkTitle] = React.useState("LoRA Benchmark");
  const [search, setSearch] = React.useState("");
  const [statusFilter, setStatusFilter] = React.useState("all");
  const [sortMode, setSortMode] = React.useState("epoch_desc");
  const [selectedPaths, setSelectedPaths] = React.useState<Set<string>>(() => new Set());
  const [presetKey, setPresetKey] = React.useState("rap");
  const selectedPreset = genres.find((item) => item.key === presetKey) ?? genres[0] ?? FALLBACK_GENRES[0];
  const [caption, setCaption] = React.useState(selectedPreset.caption_tags);
  const [lyrics, setLyrics] = React.useState(selectedPreset.lyrics || selectedPreset.test_lyrics || "");
  const [tags, setTags] = React.useState("rap, hip hop, hard drums, deep bass");
  const [negativeTags, setNegativeTags] = React.useState("generic lyrics, muddy mix, mumbled vocals");
  const [duration, setDuration] = React.useState(30);
  const [bpm, setBpm] = React.useState(Number(selectedPreset.bpm ?? 92));
  const [keyScale, setKeyScale] = React.useState(selectedPreset.keyscale || "D minor");
  const [timeSignature, setTimeSignature] = React.useState(selectedPreset.timesignature || "4/4");
  const [language, setLanguage] = React.useState("en");
  const [songModel, setSongModel] = React.useState("acestep-v15-xl-sft");
  const [audioBackend, setAudioBackend] = React.useState<AudioBackend>("mps_torch");
  const [qualityProfile, setQualityProfile] = React.useState("chart_master");
  const [inferenceSteps, setInferenceSteps] = React.useState(50);
  const [guidanceScale, setGuidanceScale] = React.useState(7);
  const [shift, setShift] = React.useState(1);
  const [seed, setSeed] = React.useState(-1);
  const [audioFormat, setAudioFormat] = React.useState("wav32");
  const [scalesText, setScalesText] = React.useState("1.0");
  const [triggerMode, setTriggerMode] = React.useState("auto");
  const [customTrigger, setCustomTrigger] = React.useState("");
  const [includeBaseline, setIncludeBaseline] = React.useState(true);
  const [stopOnError, setStopOnError] = React.useState(false);
  const [submitting, setSubmitting] = React.useState(false);
  const [activeJob, setActiveJob] = React.useState<LoraBenchmarkJob | null>(null);
  const [error, setError] = React.useState("");
  const [notes, setNotes] = React.useState<Record<string, string>>({});

  React.useEffect(() => {
    setCaption(selectedPreset.caption_tags);
    setLyrics(selectedPreset.lyrics || selectedPreset.test_lyrics || "");
    setBpm(Number(selectedPreset.bpm ?? bpm));
    setKeyScale(selectedPreset.keyscale || keyScale);
    setTimeSignature(selectedPreset.timesignature || timeSignature);
  }, [presetKey]);

  React.useEffect(() => {
    const defaults = docsCorrectDefaults(songModel);
    setInferenceSteps(defaults.inference_steps);
    setShift(defaults.shift);
  }, [songModel]);

  React.useEffect(() => {
    if (!activeJob?.id || stateDone(activeJob.state)) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const resp = await getLoraBenchmarkJob(activeJob.id);
        if (cancelled) return;
        if (resp.job) {
          setActiveJob(resp.job);
          addJob({
            id: resp.job.id,
            kind: "lora-benchmark",
            label: resp.job.benchmark_title || "LoRA Benchmark",
            progress: resp.job.progress ?? 0,
            status: resp.job.status || resp.job.stage || "running",
            state: resp.job.state || "running",
            stage: resp.job.stage,
            kindLabel: "LoRA benchmark",
            detailsPath: `/api/lora/benchmarks/jobs/${encodeURIComponent(resp.job.id)}`,
            logPath: `/api/lora/benchmarks/jobs/${encodeURIComponent(resp.job.id)}/log`,
            metadata: resp.job as unknown as Record<string, unknown>,
            error: resp.job.error || resp.job.errors?.join("\n") || "",
            startedAt: resp.job.started_at ? new Date(resp.job.started_at).getTime() : Date.now(),
            updatedAt: resp.job.updated_at,
          });
        }
      } catch (pollError) {
        if (!cancelled) setError(pollError instanceof Error ? pollError.message : "Benchmark poll failed");
      }
    };
    const interval = window.setInterval(tick, 2500);
    void tick();
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [activeJob?.id, activeJob?.state, addJob]);

  const adapters = adaptersQuery.data?.adapters ?? [];
  const filteredAdapters = React.useMemo(() => {
    const q = search.trim().toLowerCase();
    const rows = adapters.filter((adapter) => {
      const haystack = [
        loraAdapterLabel(adapter),
        adapter.path,
        adapter.model_variant,
        adapter.song_model,
        adapterQuality(adapter),
        loraTriggerOptions(adapter).join(" "),
      ].join(" ").toLowerCase();
      if (q && !haystack.includes(q)) return false;
      if (statusFilter !== "all" && adapterQuality(adapter).toLowerCase() !== statusFilter) return false;
      return true;
    });
    rows.sort((a, b) => {
      if (sortMode === "epoch_asc") return (adapterEpoch(a) ?? 0) - (adapterEpoch(b) ?? 0);
      if (sortMode === "loss_asc") return (adapterLoss(a) ?? Number.MAX_SAFE_INTEGER) - (adapterLoss(b) ?? Number.MAX_SAFE_INTEGER);
      if (sortMode === "updated_desc") return text(b.updated_at).localeCompare(text(a.updated_at));
      return (adapterEpoch(b) ?? 0) - (adapterEpoch(a) ?? 0);
    });
    return rows;
  }, [adapters, search, sortMode, statusFilter]);

  const selectedAdapters = React.useMemo(
    () => adapters.filter((adapter) => selectedPaths.has(adapter.path)),
    [adapters, selectedPaths],
  );
  const uniqueStatuses = React.useMemo(() => {
    return Array.from(new Set(adapters.map((adapter) => adapterQuality(adapter).toLowerCase()).filter(Boolean))).sort();
  }, [adapters]);
  const results = activeJob?.results ?? [];
  const bestId = activeJob?.best_manual_result_id || activeJob?.best_auto_result_id || activeJob?.best_result_id;
  const best = results.find((item) => item.attempt_id === bestId);
  const scales = parseScales(scalesText);
  const estimatedAttempts = selectedAdapters.length * Math.max(1, scales.length) + (includeBaseline ? 1 : 0);

  const toggleAdapter = (path: string, enabled: boolean) => {
    setSelectedPaths((current) => {
      const next = new Set(current);
      if (enabled) next.add(path);
      else next.delete(path);
      return next;
    });
  };

  const selectFiltered = () => {
    setSelectedPaths((current) => {
      const next = new Set(current);
      for (const adapter of filteredAdapters) next.add(adapter.path);
      return next;
    });
  };

  const clearSelected = () => setSelectedPaths(new Set());

  const startBenchmark = async () => {
    setSubmitting(true);
    setError("");
    try {
      if (!selectedAdapters.length && !includeBaseline) throw new Error("Kies adapters of zet no-LoRA baseline aan.");
      if (!scales.length) throw new Error("Vul minimaal één geldige LoRA scale in, bijvoorbeeld 1.0.");
      const body = {
        benchmark_title: benchmarkTitle,
        stop_on_error: stopOnError,
        include_baseline: includeBaseline,
        adapters: selectedAdapters,
        lora_scales: scales,
        trigger_mode: triggerMode,
        custom_trigger_tag: customTrigger,
        render_payload: {
          title: benchmarkTitle,
          artist_name: "LoRA Benchmark",
          task_type: "text2music",
          style_profile: presetKey,
          caption,
          tags,
          negative_tags: negativeTags,
          lyrics,
          instrumental: false,
          duration,
          audio_duration: duration,
          bpm,
          key_scale: keyScale,
          time_signature: timeSignature,
          vocal_language: language,
          song_model: songModel,
          audio_backend: audioBackend,
          use_mlx_dit: useMlxDitForAudioBackend(audioBackend),
          quality_profile: qualityProfile,
          inference_steps: inferenceSteps,
          guidance_scale: guidanceScale,
          shift,
          seed,
          audio_format: audioFormat,
          batch_size: 1,
          vocal_intelligibility_gate: true,
          save_to_library: false,
        },
      };
      const resp = await startLoraBenchmarkJob(body);
      if (!resp.success || !resp.job_id) throw new Error(resp.error || "LoRA Benchmark starten mislukt");
      const job = resp.job || ({ id: resp.job_id, benchmark_title: benchmarkTitle, state: "queued" } as LoraBenchmarkJob);
      setActiveJob(job);
      addJob({
        id: resp.job_id,
        kind: "lora-benchmark",
        label: benchmarkTitle || "LoRA Benchmark",
        progress: job.progress ?? 0,
        status: job.status || "queued",
        state: job.state || "queued",
        stage: job.stage,
        kindLabel: "LoRA benchmark",
        detailsPath: `/api/lora/benchmarks/jobs/${encodeURIComponent(resp.job_id)}`,
        logPath: `/api/lora/benchmarks/jobs/${encodeURIComponent(resp.job_id)}/log`,
        metadata: job as unknown as Record<string, unknown>,
        startedAt: Date.now(),
      });
      openJob(resp.job_id);
      toast.success("LoRA Benchmark gestart.");
    } catch (submitError) {
      const message = submitError instanceof Error ? submitError.message : "Benchmark starten mislukt";
      setError(message);
      toast.error(message);
    } finally {
      setSubmitting(false);
    }
  };

  const saveRating = async (attemptId: string, rating: number) => {
    if (!activeJob?.id) return;
    try {
      const resp = await rateLoraBenchmarkResult(activeJob.id, {
        attempt_id: attemptId,
        user_rating: rating,
        user_notes: notes[attemptId] || "",
      });
      if (resp.job) setActiveJob(resp.job);
      toast.success(`Luisterrating opgeslagen: ${rating}/5`);
    } catch (ratingError) {
      toast.error(ratingError instanceof Error ? ratingError.message : "Rating opslaan mislukt");
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="border-b border-border/60 px-6 py-5 sm:px-10">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div className="space-y-1">
            <p className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">Wizard</p>
            <h1 className="font-display text-2xl font-semibold sm:text-3xl">LoRA Benchmark</h1>
            <p className="max-w-3xl text-sm text-muted-foreground">
              Test alle LoRAs met dezelfde 30s lyrics, scales en renderinstellingen. Audio komt per attempt direct beschikbaar, met automatische score en jouw luisterrating.
            </p>
          </div>
          <Button onClick={startBenchmark} size="lg" disabled={submitting || estimatedAttempts === 0}>
            {submitting ? <RefreshCw className="size-4 animate-spin" /> : <Play className="size-4" />}
            Start benchmark
          </Button>
        </div>
      </header>

      <main className="min-h-0 flex-1 overflow-y-auto px-6 py-8 sm:px-10">
        <div className="mx-auto grid w-full max-w-7xl gap-6 xl:grid-cols-[340px_minmax(0,1fr)_420px]">
          <aside className="space-y-4">
            <FieldGroup title="Adapters" description="Alle adapters zijn zichtbaar; unsafe statussen krijgen badges maar blijven testbaar.">
              <div className="space-y-3">
                <div className="relative">
                  <Search className="pointer-events-none absolute left-3 top-2.5 size-4 text-muted-foreground" />
                  <Input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Zoek LoRA, epoch, trigger..." className="pl-9" />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <Select value={statusFilter} onValueChange={setStatusFilter}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">Alle statussen</SelectItem>
                      {uniqueStatuses.map((status) => (
                        <SelectItem key={status} value={status}>{status}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Select value={sortMode} onValueChange={setSortMode}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="epoch_desc">Nieuwste epoch</SelectItem>
                      <SelectItem value="epoch_asc">Oudste epoch</SelectItem>
                      <SelectItem value="loss_asc">Laagste loss</SelectItem>
                      <SelectItem value="updated_desc">Laatst gewijzigd</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex gap-2">
                  <Button type="button" variant="outline" size="sm" onClick={selectFiltered}>
                    <Filter className="size-3.5" />
                    Selecteer zichtbaar
                  </Button>
                  <Button type="button" variant="ghost" size="sm" onClick={clearSelected}>
                    Wis
                  </Button>
                </div>
                <div className="max-h-[48vh] space-y-2 overflow-y-auto pr-1">
                  {filteredAdapters.map((adapter) => {
                    const quality = adapterQuality(adapter);
                    const selected = selectedPaths.has(adapter.path);
                    const epoch = adapterEpoch(adapter);
                    const loss = adapterLoss(adapter);
                    const triggers = loraTriggerOptions(adapter);
                    const unsafe = ["quarantined", "failed_audition", "not_generation_loadable"].includes(quality.toLowerCase());
                    return (
                      <button
                        key={adapter.path}
                        type="button"
                        onClick={() => toggleAdapter(adapter.path, !selected)}
                        className={cn(
                          "w-full rounded-md border p-3 text-left transition-colors",
                          selected ? "border-primary/50 bg-primary/10" : "border-border/70 bg-background/35 hover:bg-accent/50",
                        )}
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="min-w-0">
                            <p className="truncate text-sm font-medium">{loraAdapterLabel(adapter)}</p>
                            <p className="truncate text-[11px] text-muted-foreground">{adapter.path}</p>
                          </div>
                          <Switch checked={selected} onCheckedChange={(checked) => toggleAdapter(adapter.path, checked)} onClick={(event) => event.stopPropagation()} />
                        </div>
                        <div className="mt-2 flex flex-wrap gap-1.5">
                          <Badge variant={unsafe ? "destructive" : quality === "needs_review" ? "secondary" : "outline"}>{quality}</Badge>
                          {epoch !== null && <Badge variant="outline">epoch {epoch}</Badge>}
                          {loss !== null && <Badge variant="outline">loss {loss.toFixed(3)}</Badge>}
                          {(adapter.model_variant || adapter.song_model) && <Badge variant="outline">{adapter.model_variant || adapter.song_model}</Badge>}
                          {triggers[0] && <Badge variant="outline">trigger {triggers[0]}</Badge>}
                        </div>
                      </button>
                    );
                  })}
                  {!filteredAdapters.length && (
                    <p className="rounded-md border bg-background/35 p-3 text-sm text-muted-foreground">Geen adapters gevonden voor deze filter.</p>
                  )}
                </div>
              </div>
            </FieldGroup>
          </aside>

          <section className="space-y-4">
            <FieldGroup title="Test prompt" description="Caption is productie/stijl; lyrics is het exacte tijdscript voor iedere LoRA.">
              <div className="grid gap-3 md:grid-cols-2">
                <div className="space-y-1.5">
                  <Label>Benchmark titel</Label>
                  <Input value={benchmarkTitle} onChange={(event) => setBenchmarkTitle(event.target.value)} />
                </div>
                <div className="space-y-1.5">
                  <Label>Test-WAV genre</Label>
                  <Select value={presetKey} onValueChange={setPresetKey}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {genres.map((genre) => (
                        <SelectItem key={genre.key} value={genre.key}>{genre.label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="space-y-1.5">
                <Label>Caption</Label>
                <Textarea value={caption} onChange={(event) => setCaption(event.target.value)} rows={4} />
              </div>
              <div className="grid gap-3 md:grid-cols-2">
                <div className="space-y-1.5">
                  <Label>Tags</Label>
                  <Input value={tags} onChange={(event) => setTags(event.target.value)} />
                </div>
                <div className="space-y-1.5">
                  <Label>Negative tags</Label>
                  <Input value={negativeTags} onChange={(event) => setNegativeTags(event.target.value)} />
                </div>
              </div>
              <div className="space-y-1.5">
                <Label>Lyrics</Label>
                <Textarea value={lyrics} onChange={(event) => setLyrics(event.target.value)} rows={10} className="font-mono text-xs" />
              </div>
            </FieldGroup>

            <FieldGroup title="Render settings" description="Default blijft MPS/Torch en 30 seconden, maar alle ACE-Step/LoRA benchmarkknoppen zitten hier.">
              <div className="grid gap-3 md:grid-cols-3">
                <div className="space-y-1.5">
                  <Label>Duur</Label>
                  <Input type="number" min={10} max={600} value={duration} onChange={(event) => setDuration(Number(event.target.value))} />
                </div>
                <div className="space-y-1.5">
                  <Label>BPM</Label>
                  <Input type="number" min={40} max={220} value={bpm} onChange={(event) => setBpm(Number(event.target.value))} />
                </div>
                <div className="space-y-1.5">
                  <Label>Taal</Label>
                  <Select value={language} onValueChange={setLanguage}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {ACE_STEP_LANGUAGE_OPTIONS.map(([value, label]) => (
                        <SelectItem key={value} value={value}>{label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1.5">
                  <Label>Key</Label>
                  <Select value={keyScale} onValueChange={setKeyScale}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {ACE_STEP_KEY_SCALE_OPTIONS.map((key) => (
                        <SelectItem key={key} value={key}>{key}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1.5">
                  <Label>Time signature</Label>
                  <Select value={timeSignature} onValueChange={setTimeSignature}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {ACE_STEP_TIME_SIGNATURE_OPTIONS.filter(([value]) => value).map(([value, label]) => (
                        <SelectItem key={value} value={value}>{label}</SelectItem>
                      ))}
                      <SelectItem value="4/4">4/4</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1.5">
                  <Label>Audio format</Label>
                  <Select value={audioFormat} onValueChange={setAudioFormat}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {OFFICIAL_AUDIO_FORMAT_OPTIONS.map(([value, label]) => (
                        <SelectItem key={value} value={value}>{label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1.5">
                  <Label>Model</Label>
                  <Select value={songModel} onValueChange={setSongModel}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {SONG_MODELS.map(([value, label]) => (
                        <SelectItem key={value} value={value}>{label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1.5">
                  <Label>Backend</Label>
                  <Select value={audioBackend} onValueChange={(value) => setAudioBackend(value as AudioBackend)}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="mps_torch">MPS/Torch</SelectItem>
                      <SelectItem value="mlx">MLX</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1.5">
                  <Label>Quality</Label>
                  <Select value={qualityProfile} onValueChange={setQualityProfile}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {QUALITY_PROFILES.map(([value, label]) => (
                        <SelectItem key={value} value={value}>{label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1.5">
                  <Label>Steps</Label>
                  <Input type="number" min={1} max={200} value={inferenceSteps} onChange={(event) => setInferenceSteps(Number(event.target.value))} />
                </div>
                <div className="space-y-1.5">
                  <Label>Guidance</Label>
                  <Input type="number" min={1} max={15} step={0.1} value={guidanceScale} onChange={(event) => setGuidanceScale(Number(event.target.value))} />
                </div>
                <div className="space-y-1.5">
                  <Label>Shift</Label>
                  <Input type="number" min={0} max={10} step={0.1} value={shift} onChange={(event) => setShift(Number(event.target.value))} />
                </div>
                <div className="space-y-1.5">
                  <Label>Seed</Label>
                  <Input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
                </div>
                <div className="space-y-1.5">
                  <Label>LoRA scales</Label>
                  <Input value={scalesText} onChange={(event) => setScalesText(event.target.value)} placeholder="1.0, 0.5" />
                </div>
                <div className="space-y-1.5">
                  <Label>Trigger mode</Label>
                  <Select value={triggerMode} onValueChange={setTriggerMode}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="auto">Auto uit metadata</SelectItem>
                      <SelectItem value="custom">Custom trigger</SelectItem>
                      <SelectItem value="off">Trigger uit</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                {triggerMode === "custom" && (
                  <div className="space-y-1.5">
                    <Label>Custom trigger</Label>
                    <Input value={customTrigger} onChange={(event) => setCustomTrigger(event.target.value)} />
                  </div>
                )}
              </div>
              <div className="grid gap-3 md:grid-cols-2">
                <div className="flex items-center justify-between rounded-md border bg-background/35 p-3">
                  <Label>No-LoRA baseline</Label>
                  <Switch checked={includeBaseline} onCheckedChange={setIncludeBaseline} />
                </div>
                <div className="flex items-center justify-between rounded-md border bg-background/35 p-3">
                  <Label>Stop bij eerste fout</Label>
                  <Switch checked={stopOnError} onCheckedChange={setStopOnError} />
                </div>
              </div>
              <div className="rounded-md border bg-background/35 p-3 text-sm text-muted-foreground">
                {selectedAdapters.length} adapters × {scales.length || 0} scale(s) {includeBaseline ? "+ baseline" : ""} = {estimatedAttempts} attempts.
              </div>
            </FieldGroup>
          </section>

          <aside className="space-y-4">
            <FieldGroup title="Beste kandidaat">
              {best ? (
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="size-5 text-emerald-500" />
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium">{resultLabel(best)}</p>
                      <p className="text-xs text-muted-foreground">Score {resultScore(best).toFixed(1)} · rating {best.user_rating || 0}/5</p>
                    </div>
                  </div>
                  <GenerationAudioList result={best.result || { audio_url: best.audio_urls?.[0] }} title={resultLabel(best)} artist="LoRA Benchmark" />
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">Start een benchmark; de beste kandidaat verschijnt hier zodra resultaten binnenkomen.</p>
              )}
            </FieldGroup>

            <FieldGroup title="Live voortgang">
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span>{activeJob?.status || "Nog niet gestart"}</span>
                  <span className="font-mono">{activeJob?.progress ?? 0}%</span>
                </div>
                <Progress value={activeJob?.progress ?? 0} className="h-1.5" />
                <div className="grid grid-cols-3 gap-2 text-center text-xs">
                  <div className="rounded-md border bg-background/35 p-2">
                    <p className="font-mono text-base">{activeJob?.completed_attempts ?? 0}</p>
                    <p className="text-muted-foreground">klaar</p>
                  </div>
                  <div className="rounded-md border bg-background/35 p-2">
                    <p className="font-mono text-base">{activeJob?.failed_attempts ?? 0}</p>
                    <p className="text-muted-foreground">failed</p>
                  </div>
                  <div className="rounded-md border bg-background/35 p-2">
                    <p className="font-mono text-base">{activeJob?.remaining_attempts ?? estimatedAttempts}</p>
                    <p className="text-muted-foreground">te gaan</p>
                  </div>
                </div>
                {error && <p className="rounded-md bg-destructive/10 p-2 text-xs text-destructive">{error}</p>}
              </div>
            </FieldGroup>

            <FieldGroup title="Score grafiek" className="xl:sticky xl:top-4">
              <BarGraph results={results} bestId={best?.attempt_id} />
            </FieldGroup>
          </aside>
        </div>

        <div className="mx-auto mt-6 grid w-full max-w-7xl gap-6 xl:grid-cols-[minmax(0,1fr)_420px]">
          <FieldGroup title="Resultaten en audio">
            <div className="space-y-4">
              {results.map((result, index) => {
                const score = resultScore(result);
                const attemptId = result.attempt_id || `result-${index}`;
                const breakdown = result.score_breakdown || {};
                return (
                  <div key={attemptId} className="rounded-md border bg-background/35 p-4">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="flex flex-wrap items-center gap-2">
                          <h3 className="truncate text-sm font-semibold">{resultLabel(result)}</h3>
                          {attemptId === activeJob?.best_auto_result_id && <Badge>Auto best</Badge>}
                          {attemptId === activeJob?.best_manual_result_id && <Badge variant="secondary">Luister-best</Badge>}
                          <Badge variant={String(result.state).toLowerCase() === "failed" ? "destructive" : "outline"}>{result.status || result.state}</Badge>
                        </div>
                        <p className="mt-1 text-xs text-muted-foreground">
                          trigger {result.trigger_tag || "off"} · gate {result.gate_status || text(breakdown.vocal_gate_status, "—")} · job {result.generation_job_id}
                        </p>
                      </div>
                      <div className="min-w-[130px] text-right">
                        <p className="font-mono text-xl font-semibold">{score.toFixed(1)}</p>
                        <p className="text-xs text-muted-foreground">auto score</p>
                      </div>
                    </div>
                    <div className="mt-3 h-2 overflow-hidden rounded-full bg-muted">
                      <div className="h-full bg-primary" style={{ width: `${score}%` }} />
                    </div>
                    {result.error && <p className="mt-3 rounded-md bg-destructive/10 p-2 text-xs text-destructive">{result.error}</p>}
                    <GenerationAudioList result={result.result || { audio_url: result.audio_urls?.[0] }} title={resultLabel(result)} artist="LoRA Benchmark" className="mt-4 space-y-2" />
                    <div className="mt-4 grid gap-3 md:grid-cols-[minmax(0,1fr)_220px]">
                      <div className="space-y-1.5">
                        <Label>Transcript preview / redenen</Label>
                        <p className="min-h-10 rounded-md border bg-background/50 p-2 text-xs text-muted-foreground">
                          {resultTranscript(result) || text((breakdown.reasons as string[] | undefined)?.join(", "), "Nog geen transcriptdata.")}
                        </p>
                      </div>
                      <div className="space-y-2">
                        <Label>Luisterrating</Label>
                        <div className="flex gap-1">
                          {[1, 2, 3, 4, 5].map((rating) => (
                            <Button key={rating} type="button" variant={number(result.user_rating, 0) >= rating ? "default" : "outline"} size="icon" onClick={() => void saveRating(attemptId, rating)}>
                              <Star className="size-4" />
                            </Button>
                          ))}
                        </div>
                        <Input
                          value={notes[attemptId] ?? result.user_notes ?? ""}
                          onChange={(event) => setNotes((current) => ({ ...current, [attemptId]: event.target.value }))}
                          onBlur={() => {
                            if (number(result.user_rating, 0) > 0) void saveRating(attemptId, number(result.user_rating, 0));
                          }}
                          placeholder="Korte luister-notitie"
                        />
                      </div>
                    </div>
                  </div>
                );
              })}
              {!results.length && (
                <p className="rounded-md border bg-background/35 p-4 text-sm text-muted-foreground">
                  Nog geen resultaten. Zodra attempt 1 klaar is kun je direct luisteren terwijl de rest verder rendert.
                </p>
              )}
            </div>
          </FieldGroup>

          <FieldGroup title="Data grafieken" description="Gebruik dit om loss, epoch en score naast je oren te leggen.">
            <LossScoreGraph results={results} />
          </FieldGroup>
        </div>
      </main>
    </div>
  );
}
