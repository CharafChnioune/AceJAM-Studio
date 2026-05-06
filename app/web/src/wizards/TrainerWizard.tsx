import * as React from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import {
  Loader2, Music4, Upload, GraduationCap, X, FileMusic, Mic2, SkipForward,
} from "lucide-react";

import { WizardShell, FieldGroup, type WizardStepDef } from "@/components/wizard/WizardShell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  api,
  getLoraAutolabelJob,
  startLoraAutolabelJob,
  type LoraAutolabelJob,
  type LoraAutolabelLabel,
} from "@/lib/api";
import { useJobsStore } from "@/store/jobs";
import { toast } from "@/components/ui/sonner";
import { cn } from "@/lib/utils";

interface TrainerForm {
  dataset_id: string;
  trigger_tag: string;
  genre: string;
  epoch_audition_genre: string;
  genre_ratio: number;
  default_language: string;
  default_bpm: number;
  default_time_signature: string;
  learning_rate: number;
  train_epochs: number;
  batch_size: number;
  auto_understand_music: boolean;
}

interface DatasetState {
  dataset_id: string;
  files: number;
  copied_files?: string[];
  skipped_files?: string[];
  status: string;
  health?: Record<string, unknown>;
}

interface TrainJobState {
  id: string;
  state?: string;
  status?: string;
  stage?: string;
  progress?: number;
  step?: number;
  total_steps?: number;
  current_file?: string;
  transcribe_processed?: number;
  transcribe_total?: number;
  transcribe_succeeded?: number;
  transcribe_failed?: number;
  transcribe_labels?: LoraAutolabelLabel[];
}

const ALLOWED_AUDIO = /\.(wav|mp3|flac|ogg|m4a|aac)$/i;

interface EpochAuditionGenre {
  key: string;
  label: string;
  caption_tags: string;
  lyrics: string;
  bpm?: number | null;
  keyscale?: string;
  timesignature?: string;
  default?: boolean;
}

const FALLBACK_AUDITION_GENRES: EpochAuditionGenre[] = [
  {
    key: "rap",
    label: "Rap / Hip-hop",
    caption_tags: "hip hop drums, deep bass, clear spoken-word lead vocal, steady groove",
    lyrics:
      "[Verse]\nI step to the light with the pressure on ten\nEvery bar lands clean when the drums come in\n\n[Chorus]\nHands in the air when the bassline rolls\nSay it one time and the whole room knows",
    bpm: 95,
    keyscale: "A minor",
    timesignature: "4",
  },
  {
    key: "pop",
    label: "Pop",
    caption_tags: "modern pop groove, bright hook, clean lead vocal, radio-ready drums",
    lyrics:
      "[Verse]\nCity lights are turning gold tonight\nWe chase the spark until the morning light\n\n[Chorus]\nHold on hold on we are alive\nHearts beat louder when the chorus arrives",
    bpm: 118,
    keyscale: "C major",
    timesignature: "4",
  },
  {
    key: "rnb",
    label: "Soul / R&B",
    caption_tags: "smooth rnb groove, warm keys, clean intimate lead vocal, soft harmonies",
    lyrics:
      "[Verse]\nLate night glow on the window frame\nYour voice comes close and it says my name\n\n[Chorus]\nStay right here where the rhythm is slow\nLet the whole room breathe when the candles glow",
    bpm: 82,
    keyscale: "D minor",
    timesignature: "4",
  },
  {
    key: "rock",
    label: "Rock",
    caption_tags: "driving rock drums, electric guitars, clear lead vocal, strong chorus",
    lyrics:
      "[Verse]\nRoad lights flash on the edge of town\nWe hit the floor when the walls come down\n\n[Chorus]\nRaise it up with the thunder and fire\nOne loud heart in a live wire choir",
    bpm: 128,
    keyscale: "E minor",
    timesignature: "4",
  },
  {
    key: "edm",
    label: "EDM / Dance",
    caption_tags: "electronic dance beat, pulsing synth bass, clean vocal hook, club energy",
    lyrics:
      "[Verse]\nBlue lights move when the kick comes through\nEvery heartbeat locks into the groove\n\n[Chorus]\nLift me higher when the drop arrives\nWe come alive under flashing lights",
    bpm: 124,
    keyscale: "F# minor",
    timesignature: "4",
  },
  {
    key: "cinematic",
    label: "Cinematic",
    caption_tags: "cinematic drums, wide strings, clear dramatic vocal, spacious arrangement",
    lyrics:
      "[Verse]\nStars lean close as the shadows rise\nWe hold the line under open skies\n\n[Chorus]\nStand as one when the thunder calls\nLight breaks through every ancient wall",
    bpm: 88,
    keyscale: "D minor",
    timesignature: "4",
  },
  {
    key: "country",
    label: "Country / Folk",
    caption_tags: "warm acoustic guitars, steady country drums, clear heartfelt vocal",
    lyrics:
      "[Verse]\nDust on my boots and the sun sinking low\nOne more mile down a familiar road\n\n[Chorus]\nTake me home where the porch light shines\nGood hearts gather at closing time",
    bpm: 96,
    keyscale: "G major",
    timesignature: "4",
  },
];

function labelSourceBadge(source?: string) {
  if (!source) return { variant: "muted" as const, label: "—" };
  if (source === "existing_sidecar")
    return { variant: "muted" as const, label: "bestaande sidecar" };
  if (source === "online_lyrics_ovh")
    return { variant: "default" as const, label: "online lyrics" };
  if (source === "online_lyrics_missing")
    return { variant: "muted" as const, label: "geen match" };
  if (source === "official_ace_step_understand_music")
    return { variant: "default" as const, label: "AI gelabeld" };
  if (source === "understand_music_failed")
    return { variant: "destructive" as const, label: "fout" };
  return { variant: "muted" as const, label: source };
}

function AutolabelLiveView({ job }: { job: LoraAutolabelJob }) {
  const labels = job.labels ?? [];
  const state = (job.state ?? "").toLowerCase();
  const inFlight = state !== "complete" && state !== "error";
  const succeeded = job.succeeded ?? 0;
  const failed = job.failed ?? 0;
  const total = job.total ?? 0;
  const progress = job.progress ?? 0;
  const reverseLabels = React.useMemo(
    () => [...labels].reverse(),
    [labels],
  );
  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-3 rounded-xl border border-primary/30 bg-primary/5 p-4"
    >
      <div className="flex items-center gap-3">
        <Loader2
          className={cn(
            "size-5 text-primary",
            inFlight && "animate-spin",
          )}
        />
        <div className="min-w-0 flex-1">
          <p className="truncate font-medium">
            {state === "complete"
              ? "Auto-label klaar"
              : state === "error"
                ? "Auto-label fout"
                : "AI labelt elke clip…"}
          </p>
          <p className="truncate text-[11px] text-muted-foreground">
            {job.current_file
              ? `bezig: ${job.current_file}`
              : `${job.processed ?? 0}/${total} verwerkt`}
          </p>
        </div>
        <span className="font-mono text-sm tabular-nums">{progress}%</span>
      </div>

      <Progress value={progress} />

      <div className="flex flex-wrap gap-1.5 text-[10px]">
        <Badge variant="muted">{total} totaal</Badge>
        {succeeded > 0 && (
          <Badge variant="muted" className="text-emerald-300">
            ✓ {succeeded} gelabeld
          </Badge>
        )}
        {failed > 0 && (
          <Badge variant="destructive">✗ {failed} mislukt</Badge>
        )}
        {state && (
          <Badge variant="outline" className="text-[10px]">
            state: {state}
          </Badge>
        )}
      </div>

      {labels.length > 0 && (
        <div className="space-y-1">
          <p className="text-[10px] uppercase tracking-widest text-muted-foreground">
            Live labels — meest recente eerst
          </p>
          <div className="scroll-fade-y no-scrollbar max-h-[260px] space-y-1.5 overflow-y-auto pr-1">
            <AnimatePresence initial={false}>
              {reverseLabels.map((label, idx) => (
                <LiveLabelRow key={`${label.filename}-${idx}`} label={label} />
              ))}
            </AnimatePresence>
          </div>
        </div>
      )}

      {state === "error" && job.errors && job.errors.length > 0 && (
        <details className="rounded-md border bg-background/40 p-2 text-xs">
          <summary className="cursor-pointer font-medium">
            {job.errors.length} fout(en)
          </summary>
          <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap text-[10px] text-destructive">
            {job.errors.join("\n")}
          </pre>
        </details>
      )}
    </motion.div>
  );
}

function LiveLabelRow({ label }: { label: LoraAutolabelLabel }) {
  const badge = labelSourceBadge(label.label_source);
  const lyrics = (label.lyrics ?? "").replace(/\s+/g, " ").trim();
  const lyricsPreview =
    lyrics.length > 140 ? `${lyrics.slice(0, 140)}…` : lyrics;
  const isInstrumental = lyrics.toLowerCase() === "[instrumental]";
  return (
    <motion.div
      layout
      initial={{ opacity: 0, x: -6 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -6 }}
      transition={{ duration: 0.15 }}
      className="rounded-md border bg-background/40 p-2 text-xs"
    >
      <div className="flex items-center gap-2">
        <FileMusic className="size-3.5 shrink-0 text-primary" />
        <span className="flex-1 truncate font-mono">{label.filename}</span>
        <Badge variant={badge.variant} className="text-[10px]">
          {badge.label}
        </Badge>
      </div>
      {label.caption && (
        <p className="mt-1 truncate text-[10px] text-muted-foreground">
          “{label.caption}”
        </p>
      )}
      <p
        className={cn(
          "mt-1 line-clamp-2 text-[11px] leading-snug",
          isInstrumental ? "italic text-muted-foreground" : "text-foreground/85",
        )}
      >
        {lyricsPreview || "—"}
      </p>
      <div className="mt-1 flex flex-wrap gap-1 text-[10px] text-muted-foreground">
        {label.language && (
          <span className="rounded bg-background/60 px-1.5">{label.language}</span>
        )}
        {label.bpm != null && (
          <span className="rounded bg-background/60 px-1.5">{label.bpm} bpm</span>
        )}
        {label.keyscale && (
          <span className="rounded bg-background/60 px-1.5">{label.keyscale}</span>
        )}
        {label.error && (
          <span className="rounded bg-destructive/30 px-1.5 text-destructive-foreground">
            {label.error.slice(0, 60)}
          </span>
        )}
      </div>
    </motion.div>
  );
}

export function TrainerWizard() {
  const navigate = useNavigate();
  const [step, setStep] = React.useState(0);
  const [files, setFiles] = React.useState<File[]>([]);
  const [drag, setDrag] = React.useState(false);
  const inputRef = React.useRef<HTMLInputElement>(null);
  const [dataset, setDataset] = React.useState<DatasetState | null>(null);
  const [autolabelJob, setAutolabelJob] = React.useState<LoraAutolabelJob | null>(null);
  const [autolabelSkipped, setAutolabelSkipped] = React.useState(false);
  const [job, setJob] = React.useState<TrainJobState | null>(null);
  const [form, setForm] = React.useState<TrainerForm>({
    dataset_id: "",
    trigger_tag: "",
    genre: "hip hop drums, deep bass, clear spoken-word lead vocal, steady groove",
    epoch_audition_genre: "rap",
    genre_ratio: 0,
    default_language: "en",
    default_bpm: 120,
    default_time_signature: "4/4",
    learning_rate: 1e-4,
    train_epochs: 10,
    batch_size: 1,
    auto_understand_music: true,
  });

  const addJob = useJobsStore((s) => s.addJob);
  const updateJobStore = useJobsStore((s) => s.updateJob);
  const removeJob = useJobsStore((s) => s.removeJob);
  const genreQuery = useQuery({
    queryKey: ["lora-epoch-audition-genres"],
    queryFn: () =>
      api.get<{ success: boolean; genres?: EpochAuditionGenre[] }>(
        "/api/lora/epoch-audition/genres",
      ),
    staleTime: 5 * 60 * 1000,
  });
  const auditionGenres = React.useMemo(() => {
    const fromApi = genreQuery.data?.genres?.filter((item) => item.key) ?? [];
    return fromApi.length > 0 ? fromApi : FALLBACK_AUDITION_GENRES;
  }, [genreQuery.data?.genres]);
  const selectedAuditionGenre =
    auditionGenres.find((item) => item.key === form.epoch_audition_genre) ??
    auditionGenres.find((item) => item.key === "rap") ??
    auditionGenres[0];

  const setAuditionGenre = (value: string) => {
    const selected = auditionGenres.find((item) => item.key === value);
    setForm((f) => ({
      ...f,
      epoch_audition_genre: value,
      genre: selected?.caption_tags || f.genre,
      default_bpm: typeof selected?.bpm === "number" && selected.bpm > 0 ? selected.bpm : f.default_bpm,
    }));
  };

  // ---- File selection ----------------------------------------------------

  const onPickFiles = (incoming: FileList | File[] | null) => {
    if (!incoming) return;
    const arr = Array.from(incoming).filter(
      (f) => ALLOWED_AUDIO.test(f.name) || /\.(txt|json|csv)$/i.test(f.name),
    );
    if (arr.length === 0) {
      toast.error("Geen audio-bestanden geselecteerd (wav/mp3/flac/ogg/m4a/aac)");
      return;
    }
    // Dedupe by name
    const merged = [
      ...files,
      ...arr.filter((f) => !files.some((p) => p.name === f.name)),
    ];
    setFiles(merged);
  };

  const removeFile = (idx: number) =>
    setFiles((prev) => prev.filter((_, i) => i !== idx));

  // ---- Dataset import (multipart upload) ---------------------------------

  const importMutation = useMutation({
    mutationFn: async () => {
      const fd = new FormData();
      for (const f of files) fd.append("files", f);
      if (form.dataset_id) fd.append("dataset_id", form.dataset_id);
      if (form.trigger_tag) fd.append("trigger_tag", form.trigger_tag);
      if (form.default_language) fd.append("language", form.default_language);
      return api.post<{
        success: boolean;
        dataset_id?: string;
        copied_files?: string[];
        skipped_files?: string[];
        files?: unknown[];
        error?: string;
      }>("/api/lora/dataset/import-folder", fd);
    },
    onSuccess: (resp) => {
      if (!resp.success || !resp.dataset_id) {
        toast.error(resp.error || "Import mislukt");
        return;
      }
      const datasetId = resp.dataset_id;
      const fileCount = resp.copied_files?.length ?? 0;
      setDataset({
        dataset_id: datasetId,
        files: fileCount,
        copied_files: resp.copied_files,
        skipped_files: resp.skipped_files,
        status: "imported",
      });
      setForm((f) => ({ ...f, dataset_id: datasetId }));
      toast.success(
        `${fileCount} audio-bestanden geïmporteerd. AI labeling gebeurt zodra je op Start training drukt.`,
      );
      // Note: we deliberately do NOT auto-start the understand_music job
      // here. ACE-Step's LM is heavy (loads the 5Hz LM + extracts codes per
      // file) so we run it only at training time as the `transcribe` stage
      // of `_run_one_click_job`. Users who want to label upfront can press
      // "Run autolabel-job nu" on the next wizard step.
    },
    onError: (e: Error) => toast.error(e.message),
  });

  // ---- AI auto-label (understand_music) job ------------------------------

  const startAutolabel = useMutation({
    mutationFn: (datasetId?: string) => {
      const id = (datasetId ?? dataset?.dataset_id ?? "").trim();
      if (!id) throw new Error("Geen dataset_id beschikbaar");
      return startLoraAutolabelJob({
        dataset_id: id,
        language: form.default_language,
        skip_existing: true,
      });
    },
    onSuccess: (resp, variables) => {
      if (!resp.success || !resp.job_id) {
        toast.error(resp.error || "Auto-label kon niet starten");
        return;
      }
      const id = resp.job_id;
      const initial: LoraAutolabelJob = resp.job ?? {
        id,
        state: "queued",
        progress: 0,
        processed: 0,
        total: 0,
      };
      setAutolabelJob(initial);
      const datasetId = variables ?? dataset?.dataset_id ?? "";
      addJob({
        id,
        kind: "lora",
        label: `Auto-label ${datasetId.slice(0, 12)}`,
        progress: 0,
        status: "queued",
        startedAt: Date.now(),
      });
    },
    onError: (e: Error) => toast.error(e.message),
  });

  React.useEffect(() => {
    const id = autolabelJob?.id;
    if (!id) return;
    const state = (autolabelJob?.state ?? "").toLowerCase();
    if (state === "complete" || state === "error") return;
    let cancelled = false;
    const tick = async () => {
      try {
        const resp = await getLoraAutolabelJob(id);
        if (cancelled) return;
        const j = resp.job;
        if (!j) return;
        setAutolabelJob(j);
        const s = (j.state ?? "running").toLowerCase();
        const desc = j.status ?? s;
        updateJobStore(id, { progress: j.progress ?? 0, status: desc });
        if (s === "complete") {
          toast.success(
            `Auto-label klaar: ${j.succeeded ?? 0} succes, ${j.failed ?? 0} mislukt`,
          );
          updateJobStore(id, { status: "complete", progress: 100 });
          setTimeout(() => removeJob(id), 4000);
          return;
        }
        if (s === "error") {
          toast.error(j.status || "Auto-label mislukt");
          updateJobStore(id, { status: "error" });
          setTimeout(() => removeJob(id), 6000);
          return;
        }
        setTimeout(tick, 2000);
      } catch (e) {
        if (!cancelled) toast.error(`Poll-fout: ${(e as Error).message}`);
      }
    };
    tick();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autolabelJob?.id]);

  // ---- Training job ------------------------------------------------------

  const startTrain = useMutation({
    mutationFn: () =>
      api.post<{
        success: boolean;
        job?: { id: string };
        error?: string;
      }>("/api/lora/one-click-train", {
        dataset_id: dataset?.dataset_id,
        trigger_tag: form.trigger_tag,
        language: form.default_language,
        genre: form.genre,
        genre_ratio: form.genre_ratio,
        epoch_audition_genre: form.epoch_audition_genre,
        epoch_audition_bpm: selectedAuditionGenre?.bpm ?? undefined,
        epoch_audition_keyscale: selectedAuditionGenre?.keyscale ?? undefined,
        epoch_audition_timesignature: selectedAuditionGenre?.timesignature ?? undefined,
        auto_understand_music: form.auto_understand_music,
        training_defaults: {
          learning_rate: form.learning_rate,
          train_epochs: form.train_epochs,
          batch_size: form.batch_size,
        },
      }),
    onSuccess: (resp) => {
      if (!resp.success || !resp.job?.id) {
        toast.error(resp.error || "Training kon niet starten");
        return;
      }
      const id = resp.job.id;
      setJob({ id, state: "queued", progress: 0 });
      addJob({
        id,
        kind: "lora",
        label: form.trigger_tag || dataset?.dataset_id || "LoRA training",
        progress: 0,
        status: "queued",
        state: "queued",
        stage: "queued",
        kindLabel: "LoRA training",
        detailsPath: `/api/lora/jobs/${encodeURIComponent(id)}`,
        logPath: `/api/lora/jobs/${encodeURIComponent(id)}/log`,
        metadata: {
          dataset_id: dataset?.dataset_id,
          trigger_tag: form.trigger_tag,
          language: form.default_language,
          epoch_audition_genre: form.epoch_audition_genre,
          learning_rate: form.learning_rate,
          train_epochs: form.train_epochs,
          batch_size: form.batch_size,
        },
        startedAt: Date.now(),
      });
      setStep(steps.length - 1);
    },
    onError: (e: Error) => toast.error(e.message),
  });

  React.useEffect(() => {
    if (!job?.id) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const resp = await api.get<{
          success: boolean;
          job?: {
            id?: string;
            kind?: string;
            state?: string;
            status?: string;
            stage?: string;
            progress?: number;
            step?: number;
            total_steps?: number;
            current_file?: string;
            transcribe_processed?: number;
            transcribe_total?: number;
            transcribe_succeeded?: number;
            transcribe_failed?: number;
            error?: string;
            created_at?: string;
            updated_at?: string;
            params?: Record<string, unknown>;
            paths?: Record<string, unknown>;
            result?: Record<string, unknown>;
          };
        }>(`/api/lora/jobs/${encodeURIComponent(job.id)}`);
        if (cancelled) return;
        const j = resp.job;
        if (!j) return;
        const state = (j.state ?? "running").toLowerCase();
        const description = j.status ?? state;
        const p = typeof j.progress === "number" ? j.progress : 0;
        // _set_job_state in lora_trainer.py merges arbitrary kwargs into
        // `result`, so the per-file transcribe progress lives there. Read
        // both top-level (legacy) and `result.*` (current) shapes.
        const r = (j.result ?? {}) as Record<string, unknown>;
        const pickNum = (k: string) => {
          const v = (j as Record<string, unknown>)[k] ?? r[k];
          return typeof v === "number" ? v : undefined;
        };
        const pickStr = (k: string) => {
          const v = (j as Record<string, unknown>)[k] ?? r[k];
          return typeof v === "string" ? v : undefined;
        };
        const labelsRaw = r["transcribe_labels"];
        const transcribeLabels = Array.isArray(labelsRaw)
          ? (labelsRaw as LoraAutolabelLabel[])
          : undefined;
        setJob({
          id: job.id,
          state,
          status: description,
          stage: j.stage,
          progress: p,
          step: j.step,
          total_steps: j.total_steps,
          current_file: pickStr("current_file"),
          transcribe_processed: pickNum("transcribe_processed"),
          transcribe_total: pickNum("transcribe_total"),
          transcribe_succeeded: pickNum("transcribe_succeeded"),
          transcribe_failed: pickNum("transcribe_failed"),
          transcribe_labels: transcribeLabels,
        });
        updateJobStore(job.id, {
          progress: p,
          status: description,
          state,
          stage: j.stage,
          updatedAt: j.updated_at || Date.now(),
          detailsPath: `/api/lora/jobs/${encodeURIComponent(job.id)}`,
          logPath: `/api/lora/jobs/${encodeURIComponent(job.id)}/log`,
          metadata: j as Record<string, unknown>,
          error: j.error,
        });
        if (state === "complete" || state === "succeeded") {
          toast.success("LoRA training klaar.");
          updateJobStore(job.id, {
            status: "complete",
            state,
            progress: 100,
            metadata: j as Record<string, unknown>,
          });
          return;
        }
        if (state === "error" || state === "failed") {
          toast.error(j.error ?? "Training mislukt");
          updateJobStore(job.id, {
            status: "error",
            state,
            error: j.error,
            metadata: j as Record<string, unknown>,
          });
          return;
        }
        setTimeout(tick, 4000);
      } catch (e) {
        if (!cancelled) toast.error(`Poll-fout: ${(e as Error).message}`);
      }
    };
    tick();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.id]);

  // ---- Steps -------------------------------------------------------------

  const steps: WizardStepDef[] = [
    {
      key: "intro",
      title: "Wat is een LoRA?",
      description:
        "LoRA = Low-Rank Adapter. Een klein bijgetraind bestand dat ACE-Step in jouw stem, jouw stijl, of jouw genre laat zingen — zonder het basismodel te herhalen.",
      isValid: true,
      render: () => (
        <div className="rounded-xl border bg-card/40 p-5 text-sm text-muted-foreground">
          <ul className="list-inside list-disc space-y-1">
            <li>
              Wijs een trigger-tag aan (bv.{" "}
              <code className="rounded bg-background/40 px-1">char_aurora</code>)
              zodat je hem later kunt activeren.
            </li>
            <li>Verzamel ~20–100 audio-bestanden in dezelfde stijl/stem.</li>
            <li>De server labelt elke clip automatisch — geen handwerk nodig.</li>
            <li>Training duurt ~10–60 min, afhankelijk van GPU en epoch-aantal.</li>
          </ul>
        </div>
      ),
    },
    {
      key: "files",
      title: "Audio-bestanden uploaden",
      description:
        "Sleep je audio hierheen of klik om te bladeren. Optioneel: voeg .txt/.json/.csv sidecar-files toe voor handmatige labels.",
      isValid: !!dataset?.dataset_id,
      render: () => (
        <div className="space-y-4">
          <input
            ref={inputRef}
            type="file"
            multiple
            accept=".wav,.mp3,.flac,.ogg,.m4a,.aac,.txt,.json,.csv,audio/*"
            className="hidden"
            onChange={(e) => onPickFiles(e.target.files)}
          />

          <FieldGroup>
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1.5">
                <Label>Trigger tag</Label>
                <Input
                  value={form.trigger_tag}
                  onChange={(e) =>
                    setForm((f) => ({ ...f, trigger_tag: e.target.value }))
                  }
                  placeholder="char_aurora"
                />
              </div>
              <div className="space-y-1.5">
                <Label>Default taal</Label>
                <Input
                  value={form.default_language}
                  onChange={(e) =>
                    setForm((f) => ({ ...f, default_language: e.target.value }))
                  }
                />
              </div>
            </div>
          </FieldGroup>

          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              onPickFiles(e.dataTransfer.files);
            }}
            onClick={() => inputRef.current?.click()}
            role="button"
            tabIndex={0}
            className={cn(
              "flex cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed bg-card/30 p-10 text-center transition-colors hover:border-primary/40 hover:bg-card/50",
              drag && "border-primary bg-primary/5",
            )}
          >
            <div className="flex size-12 items-center justify-center rounded-full bg-primary/15 text-primary">
              <Upload className="size-5" />
            </div>
            <div className="space-y-1">
              <p className="font-medium">Sleep audio hierheen</p>
              <p className="text-xs text-muted-foreground">
                of klik om te bladeren — wav, mp3, flac, ogg, m4a, aac
              </p>
            </div>
          </motion.div>

          {files.length > 0 && (
            <FieldGroup
              title={`${files.length} bestand${files.length === 1 ? "" : "en"} geselecteerd`}
            >
              <div className="max-h-48 space-y-1 overflow-y-auto pr-1">
                <AnimatePresence initial={false}>
                  {files.map((f, idx) => (
                    <motion.div
                      key={f.name + idx}
                      initial={{ opacity: 0, x: -4 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -4 }}
                      className="flex items-center gap-2 rounded-md border bg-background/40 px-2 py-1.5 text-xs"
                    >
                      <FileMusic className="size-3.5 shrink-0 text-primary" />
                      <span className="flex-1 truncate font-mono">{f.name}</span>
                      <span className="text-[10px] text-muted-foreground">
                        {(f.size / 1024 / 1024).toFixed(1)} MB
                      </span>
                      <Button
                        size="icon-sm"
                        variant="ghost"
                        onClick={(e) => {
                          e.stopPropagation();
                          removeFile(idx);
                        }}
                      >
                        <X className="size-3" />
                      </Button>
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
              <Button
                onClick={() => importMutation.mutate()}
                disabled={
                  files.length === 0 || !form.trigger_tag || importMutation.isPending
                }
                className="w-full gap-2"
              >
                {importMutation.isPending ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <Upload className="size-4" />
                )}
                {importMutation.isPending
                  ? "Uploaden + auto-labelen…"
                  : "Importeer + label dataset"}
              </Button>
            </FieldGroup>
          )}

          {dataset?.dataset_id && (
            <motion.div
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground"
            >
              <Badge variant="muted">id: {dataset.dataset_id.slice(0, 14)}…</Badge>
              <Badge variant="muted">{dataset.files} files</Badge>
              {dataset.skipped_files && dataset.skipped_files.length > 0 && (
                <Badge variant="muted">{dataset.skipped_files.length} skipped</Badge>
              )}
              <Badge>status: {dataset.status}</Badge>
            </motion.div>
          )}

          {autolabelJob && <AutolabelLiveView job={autolabelJob} />}
        </div>
      ),
    },
    {
      key: "content",
      title: "Genre, content & auto-transcribe",
      description:
        "Vertel de trainer welk genre het is en of we vlak voor de training online lyrics moeten ophalen + ACE-Step section-tags moeten toevoegen voor elke clip die nog geen sidecars heeft.",
      isValid: !!dataset?.dataset_id,
      hidden: !dataset?.dataset_id,
      render: () => (
        <div className="space-y-4">
          <FieldGroup
            title="Test-WAV genre"
            description="Kies welke veilige testtekst en tags de epoch-audition gebruikt. Rap krijgt rap-lyrics/tags, pop krijgt pop-lyrics/tags, soul krijgt soul/R&B-lyrics/tags."
          >
            <div className="grid gap-3 lg:grid-cols-[260px_1fr_180px]">
              <div className="space-y-1.5">
                <Label>Test-WAV stijl</Label>
                <Select value={form.epoch_audition_genre} onValueChange={setAuditionGenre}>
                  <SelectTrigger>
                    <SelectValue placeholder="Kies genre" />
                  </SelectTrigger>
                  <SelectContent>
                    {auditionGenres.map((item) => (
                      <SelectItem key={item.key} value={item.key}>
                        {item.label || item.key}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label>Tags die naar de test-WAV gaan</Label>
                <Input
                  value={form.genre}
                  onChange={(e) => setForm((f) => ({ ...f, genre: e.target.value }))}
                  placeholder="bv. synthwave, drill, neo-soul"
                />
              </div>
              <div className="space-y-2">
                <div className="flex items-baseline justify-between">
                  <Label>Genre-ratio</Label>
                  <span className="font-mono text-xs">{form.genre_ratio}%</span>
                </div>
                <Slider
                  value={[form.genre_ratio]}
                  min={0}
                  max={100}
                  step={5}
                  onValueChange={(v) =>
                    setForm((f) => ({ ...f, genre_ratio: v[0] ?? 0 }))
                  }
                />
                <p className="text-[10px] text-muted-foreground">
                  Hoeveel % van de samples krijgt expliciet de genre-tag.
                </p>
              </div>
            </div>
            {selectedAuditionGenre && (
              <div className="grid gap-3 rounded-lg border bg-background/35 p-3 text-xs lg:grid-cols-[1fr_1.2fr]">
                <div className="space-y-2">
                  <div className="flex flex-wrap gap-1.5">
                    <Badge variant="muted">{selectedAuditionGenre.label}</Badge>
                    {selectedAuditionGenre.bpm && (
                      <Badge variant="outline">{selectedAuditionGenre.bpm} bpm</Badge>
                    )}
                    {selectedAuditionGenre.keyscale && (
                      <Badge variant="outline">{selectedAuditionGenre.keyscale}</Badge>
                    )}
                    {selectedAuditionGenre.timesignature && (
                      <Badge variant="outline">{selectedAuditionGenre.timesignature}/4</Badge>
                    )}
                  </div>
                  <p className="leading-relaxed text-muted-foreground">
                    {selectedAuditionGenre.caption_tags || "Auto gebruikt de dataset-caption om een passende test te kiezen."}
                  </p>
                </div>
                <pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded-md bg-black/25 p-3 text-[11px] leading-relaxed text-foreground/85">
                  {selectedAuditionGenre.lyrics || "Auto: MLX Media kiest rap/pop/soul/rock/EDM/cinematic/country op basis van je dataset."}
                </pre>
              </div>
            )}
          </FieldGroup>

          <FieldGroup
            title="Auto-label vóór training"
            description="Bij Start training leest de trainer de ID3-tags van elke clip, zoekt de echte lyrics op via lyrics.ovh en plakt er ACE-Step section-tags ([Verse N]/[Chorus]/[Bridge]) op. Snel (geen LM, alleen HTTP), en geen kapotte transcribe-output. Sidecars die al bestaan worden overgeslagen."
          >
            <div className="flex items-start justify-between gap-3 rounded-md border bg-card/40 p-3">
              <div className="min-w-0 flex-1 space-y-1">
                <Label className="flex items-center gap-1.5">
                  <Mic2 className="size-3.5" /> Online lyrics + ACE-Step section-tags
                </Label>
                <p className="text-xs text-muted-foreground">
                  Aan = vocal LoRA's krijgen echte lyric-conditioning, opgehaald
                  van lyrics.ovh op basis van ID3 artist/title (of filename als
                  fallback) met automatische{" "}
                  <code className="rounded bg-background/40 px-1">[Verse N]</code>
                  /<code className="rounded bg-background/40 px-1">[Chorus]</code>
                  /<code className="rounded bg-background/40 px-1">[Bridge]</code>{" "}
                  tagging. Uit = alle samples blijven{" "}
                  <code className="rounded bg-background/40 px-1">[Instrumental]</code>
                  (voor pure instrumentale datasets).
                </p>
              </div>
              <Button
                size="sm"
                variant={form.auto_understand_music ? "default" : "outline"}
                onClick={() =>
                  setForm((f) => ({
                    ...f,
                    auto_understand_music: !f.auto_understand_music,
                  }))
                }
                className="gap-1.5"
              >
                {form.auto_understand_music ? (
                  <>
                    <Mic2 className="size-3.5" /> Aan
                  </>
                ) : (
                  <>
                    <SkipForward className="size-3.5" /> Uit
                  </>
                )}
              </Button>
            </div>
            {!form.auto_understand_music && (
              <div className="rounded-md border border-yellow-500/30 bg-yellow-500/5 p-2.5 text-xs text-yellow-200">
                Auto-label staat uit. Alle samples blijven{" "}
                <code className="rounded bg-background/40 px-1">[Instrumental]</code>{" "}
                — alleen kiezen voor pure-instrumentale datasets.
              </div>
            )}
          </FieldGroup>
        </div>
      ),
    },
    {
      key: "train",
      title: "Training-instellingen",
      isValid: !!dataset?.dataset_id,
      render: () => (
        <FieldGroup title="Hyperparameters">
          <div className="grid gap-4 sm:grid-cols-3">
            <div className="space-y-3">
              <div className="flex items-baseline justify-between">
                <Label>Learning rate</Label>
                <span className="font-mono text-xs">
                  {form.learning_rate.toExponential(1)}
                </span>
              </div>
              <Slider
                value={[Math.log10(form.learning_rate * 1e6)]}
                min={0}
                max={4}
                step={0.5}
                onValueChange={(v) =>
                  setForm((f) => ({
                    ...f,
                    learning_rate: Math.pow(10, (v[0] ?? 2) - 6),
                  }))
                }
              />
            </div>
            <div className="space-y-3">
              <div className="flex items-baseline justify-between">
                <Label>Epochs</Label>
                <span className="font-mono text-xs">{form.train_epochs}</span>
              </div>
              <Slider
                value={[form.train_epochs]}
                min={1}
                max={50}
                step={1}
                onValueChange={(v) =>
                  setForm((f) => ({ ...f, train_epochs: v[0] ?? 10 }))
                }
              />
            </div>
            <div className="space-y-3">
              <div className="flex items-baseline justify-between">
                <Label>Batch size</Label>
                <span className="font-mono text-xs">{form.batch_size}</span>
              </div>
              <Slider
                value={[form.batch_size]}
                min={1}
                max={8}
                step={1}
                onValueChange={(v) =>
                  setForm((f) => ({ ...f, batch_size: v[0] ?? 1 }))
                }
              />
            </div>
          </div>
        </FieldGroup>
      ),
    },
    {
      key: "training",
      title: "Training-job",
      description: "Hou deze tab open totdat de status 'complete' is.",
      isValid: !job || job.state === "complete",
      hidden: !job,
      render: () => (
        <div className="space-y-3">
          <div className="rounded-xl border border-primary/30 bg-primary/5 p-5 text-sm">
            <div className="flex items-center gap-3">
              <Loader2
                className={
                  job?.state === "complete" || job?.state === "error"
                    ? "size-5 text-primary"
                    : "size-5 animate-spin text-primary"
                }
              />
              <div className="min-w-0 flex-1">
                <p className="truncate font-medium">{job?.status ?? job?.state ?? "—"}</p>
                <p className="truncate text-xs text-muted-foreground">
                  {job?.stage ? `stage: ${job.stage}` : `job ${job?.id ?? ""}`}
                  {job?.step != null && job?.total_steps != null
                    ? ` · step ${job.step}/${job.total_steps}`
                    : ""}
                </p>
              </div>
              <span className="font-mono text-sm tabular-nums">{job?.progress ?? 0}%</span>
            </div>
            <Progress value={job?.progress ?? 0} className="mt-3" />
            <div className="mt-3 flex flex-wrap gap-1.5">
              {job?.stage && (
                <Badge variant="outline" className="text-[10px]">{job.stage}</Badge>
              )}
              {typeof job?.transcribe_total === "number" && job.transcribe_total > 0 && (
                <Badge variant="muted" className="text-[10px]">
                  transcribe {job.transcribe_processed ?? 0}/{job.transcribe_total}
                </Badge>
              )}
              {typeof job?.transcribe_succeeded === "number" && job.transcribe_succeeded > 0 && (
                <Badge variant="muted" className="text-[10px]">
                  ✓ {job.transcribe_succeeded}
                </Badge>
              )}
              {typeof job?.transcribe_failed === "number" && job.transcribe_failed > 0 && (
                <Badge variant="destructive" className="text-[10px]">
                  ✗ {job.transcribe_failed}
                </Badge>
              )}
            </div>
            {job?.current_file && (
              <p className="mt-2 truncate font-mono text-[10px] text-muted-foreground">
                {job.current_file}
              </p>
            )}
            <div className="mt-4 flex flex-wrap gap-2">
              <Button
                variant="outline"
                onClick={() => navigate("/settings")}
                className="gap-2"
              >
                <Music4 className="size-4" /> Bekijk LoRA in Settings
              </Button>
            </div>
          </div>

          {/* Live per-file labels while the training thread is in the
              `transcribe` stage. Reuses the same AutolabelLiveView the
              standalone autolabel-job uses, so users get the same
              filename + lyrics-preview + caption stream. */}
          {job?.stage === "transcribe" && (
            <AutolabelLiveView
              job={{
                id: job.id,
                state: job.state,
                status: job.status,
                progress: job.progress,
                processed: job.transcribe_processed,
                total: job.transcribe_total,
                succeeded: job.transcribe_succeeded,
                failed: job.transcribe_failed,
                current_file: job.current_file,
                labels: job.transcribe_labels,
              }}
            />
          )}
        </div>
      ),
    },
  ];

  return (
    <WizardShell
      title="LoRA Trainer"
      subtitle="Train een persoonlijke ACE-Step adapter op je eigen audio-corpus."
      steps={steps}
      step={step}
      onStepChange={setStep}
      onFinish={() => startTrain.mutate()}
      isFinishing={
        startTrain.isPending ||
        (job ? job.state !== "complete" && job.state !== "error" : false)
      }
      finishLabel={
        job
          ? "Training loopt…"
          : startTrain.isPending
            ? "Job start…"
            : "Start training"
      }
    />
  );
}
