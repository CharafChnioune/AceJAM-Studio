import * as React from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  AlertTriangle,
  Brain,
  CheckCircle2,
  Disc3,
  Download,
  ExternalLink,
  GraduationCap,
  Loader2,
  Music4,
  PauseCircle,
  PlayCircle,
  RefreshCw,
  ScrollText,
  X,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import { toast } from "@/components/ui/sonner";
import { WaveformPlayer } from "@/components/audio/WaveformPlayer";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useJobsStore, type JobEntry, type JobKind } from "@/store/jobs";

const ICONS: Record<JobKind, React.ComponentType<{ className?: string }>> = {
  generation: Music4,
  album: Disc3,
  lora: GraduationCap,
  "ollama-pull": Brain,
  "model-download": Download,
  "lm-download": Download,
};

type JsonRecord = Record<string, unknown>;

const TERMINAL_STATES = new Set([
  "complete",
  "completed",
  "succeeded",
  "success",
  "error",
  "failed",
  "stopped",
]);

const ACTIVE_STATES = new Set(["queued", "running", "stopping"]);

function asRecord(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonRecord)
    : {};
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function text(value: unknown, fallback = "—"): string {
  if (value === null || value === undefined || value === "") return fallback;
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : fallback;
  if (typeof value === "boolean") return value ? "yes" : "no";
  return String(value);
}

function shortPath(value: unknown): string {
  const raw = text(value, "");
  if (!raw) return "—";
  const parts = raw.split("/");
  return parts.length > 4 ? `…/${parts.slice(-3).join("/")}` : raw;
}

function stateOf(job: JobEntry, remote?: JsonRecord): string {
  return String(remote?.state || job.state || job.status || "queued").toLowerCase();
}

function stageOf(job: JobEntry, remote?: JsonRecord): string {
  return text(remote?.stage || job.stage || remote?.status || job.status, "");
}

function progressOf(job: JobEntry, remote?: JsonRecord): number {
  const raw = remote?.progress ?? job.progress ?? 0;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? Math.max(0, Math.min(100, parsed)) : 0;
}

function isDone(job: JobEntry, remote?: JsonRecord): boolean {
  return TERMINAL_STATES.has(stateOf(job, remote)) || TERMINAL_STATES.has(String(job.status || "").toLowerCase());
}

function isActiveJob(job: JsonRecord): boolean {
  return ACTIVE_STATES.has(String(job.state || "").toLowerCase());
}

function formatTime(value: unknown): string {
  if (!value) return "—";
  const date = typeof value === "number" ? new Date(value) : new Date(String(value));
  if (Number.isNaN(date.getTime())) return text(value);
  return date.toLocaleString(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function kindLabel(kind: JobKind): string {
  if (kind === "generation") return "Song render";
  if (kind === "lora") return "LoRA training";
  if (kind === "album") return "Album job";
  if (kind === "ollama-pull") return "Ollama pull";
  if (kind === "model-download") return "Model download";
  if (kind === "lm-download") return "LM download";
  return "Background job";
}

function jobPatchFromLora(job: JsonRecord): JobEntry {
  const params = asRecord(job.params);
  const result = asRecord(job.result);
  return {
    id: text(job.id, "job"),
    kind: "lora",
    label: text(params.trigger_tag || result.display_name || job.id, "LoRA training"),
    progress: progressOf({ id: text(job.id), kind: "lora", label: "", startedAt: Date.now() }, job),
    status: stageOf({ id: text(job.id), kind: "lora", label: "", startedAt: Date.now() }, job) || stateOf({ id: text(job.id), kind: "lora", label: "", startedAt: Date.now() }, job),
    state: text(job.state, "queued"),
    stage: text(job.stage, ""),
    kindLabel: "LoRA training",
    detailsPath: `/api/lora/jobs/${encodeURIComponent(text(job.id))}`,
    logPath: `/api/lora/jobs/${encodeURIComponent(text(job.id))}/log`,
    updatedAt: text(job.updated_at, ""),
    metadata: job,
    error: text(job.error, ""),
    startedAt: job.created_at ? new Date(String(job.created_at)).getTime() : Date.now(),
  };
}

function jobPatchFromGeneration(job: JsonRecord): JobEntry {
  const summary = asRecord(job.payload_summary);
  const result = asRecord(job.result_summary);
  const title = text(result.title || summary.title || summary.caption || job.id, "Song render");
  return {
    id: text(job.id || job.task_id, "generation-job"),
    kind: "generation",
    label: title,
    progress: progressOf({ id: text(job.id), kind: "generation", label: "", startedAt: Date.now() }, job),
    status: stageOf({ id: text(job.id), kind: "generation", label: "", startedAt: Date.now() }, job) || stateOf({ id: text(job.id), kind: "generation", label: "", startedAt: Date.now() }, job),
    state: text(job.state, "queued"),
    stage: text(job.stage || job.status, ""),
    kindLabel: "Song render",
    detailsPath: `/api/generation/jobs/${encodeURIComponent(text(job.id || job.task_id))}`,
    logPath: `/api/generation/jobs/${encodeURIComponent(text(job.id || job.task_id))}/log`,
    updatedAt: text(job.updated_at, ""),
    metadata: job,
    error: text(job.error, ""),
    startedAt: job.created_at ? new Date(String(job.created_at)).getTime() : Date.now(),
  };
}

function InfoRow({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="min-w-0 rounded-md border bg-background/35 px-3 py-2">
      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</p>
      <p className="mt-1 truncate font-mono text-xs text-foreground" title={text(value)}>
        {text(value)}
      </p>
    </div>
  );
}

function Section({
  title,
  children,
  icon: Icon,
}: {
  title: string;
  children: React.ReactNode;
  icon?: React.ComponentType<{ className?: string }>;
}) {
  return (
    <section className="rounded-lg border bg-card/45 p-4">
      <div className="mb-3 flex items-center gap-2">
        {Icon && <Icon className="size-4 text-primary" />}
        <h3 className="text-sm font-semibold">{title}</h3>
      </div>
      {children}
    </section>
  );
}

function LoraDetails({
  job,
  log,
}: {
  job: JsonRecord;
  log: string;
}) {
  const params = asRecord(job.params);
  const paths = asRecord(job.paths);
  const result = asRecord(job.result);
  const warnings = asRecord(result.dataset_warnings);
  const warningList = asArray(warnings.warnings).map((item) => text(item)).filter(Boolean);
  const audition = asRecord(params.epoch_audition);
  const auditions = asArray(result.epoch_auditions).map(asRecord);

  return (
    <div className="space-y-4">
      <Section title="Training setup" icon={GraduationCap}>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          <InfoRow label="Trigger" value={params.trigger_tag} />
          <InfoRow label="Dataset" value={params.dataset_id} />
          <InfoRow label="Samples" value={result.sample_count} />
          <InfoRow label="Epochs" value={result.epochs || params.train_epochs || params.epochs} />
          <InfoRow label="Model" value={params.song_model || params.model_variant} />
          <InfoRow label="Device" value={`${text(params.device)} / ${text(params.precision)}`} />
          <InfoRow label="Batch" value={params.train_batch_size || params.batch_size} />
          <InfoRow label="Learning rate" value={params.learning_rate} />
          <InfoRow label="Rank / alpha" value={`${text(params.rank)} / ${text(params.alpha)}`} />
        </div>
      </Section>

      {(warningList.length > 0 || Boolean(warnings.vocal_audition_unreliable)) && (
        <Section title="Dataset warnings" icon={AlertTriangle}>
          <div className="space-y-2 text-xs">
            <Badge variant={warnings.vocal_audition_unreliable ? "destructive" : "outline"}>
              missing lyrics: {text(warnings.missing_lyrics_count, "0")} / {text(warnings.sample_count, "0")}
            </Badge>
            {warningList.map((item, index) => (
              <p key={index} className="rounded-md bg-destructive/10 p-2 text-destructive">
                {item}
              </p>
            ))}
          </div>
        </Section>
      )}

      <Section title="Epoch audition">
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          <InfoRow label="Enabled" value={audition.enabled} />
          <InfoRow label="Genre" value={audition.genre_profile || audition.genre} />
          <InfoRow label="Duration" value={`${text(audition.duration, "20")}s`} />
          <InfoRow label="Scale" value={audition.scale} />
          <InfoRow label="Lyrics source" value={audition.lyrics_source} />
          <InfoRow label="Caption" value={audition.caption} />
        </div>
        {auditions.length > 0 && (
          <div className="mt-3 space-y-2">
            {auditions.slice(-6).map((item, index) => (
              <div key={index} className="rounded-md border bg-background/35 p-2 text-xs">
                <div className="flex items-center justify-between gap-2">
                  <span>Epoch {text(item.epoch)}</span>
                  <Badge variant={text(item.success, "false") === "yes" ? "default" : "outline"}>
                    {text(item.status || item.state || item.success)}
                  </Badge>
                </div>
                {Boolean(item.audio_url) && (
                  <p className="mt-1 truncate font-mono text-[10px] text-muted-foreground">
                    {text(item.audio_url)}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
      </Section>

      <Section title="Outputs">
        <div className="grid gap-2 sm:grid-cols-2">
          <InfoRow label="Final adapter" value={shortPath(result.final_adapter || paths.final_adapter)} />
          <InfoRow label="Registered adapter" value={shortPath(result.registered_adapter_path)} />
          <InfoRow label="Adapter name" value={result.adapter_name} />
          <InfoRow label="Training output" value={shortPath(paths.output_dir)} />
        </div>
      </Section>

      <Section title="Log tail" icon={ScrollText}>
        <pre className="max-h-72 overflow-auto rounded-md bg-black/40 p-3 text-[11px] leading-relaxed text-muted-foreground">
          {log || "Nog geen logregels beschikbaar."}
        </pre>
      </Section>
    </div>
  );
}

function GenerationDetails({
  job,
  log,
}: {
  job: JsonRecord;
  log: string;
}) {
  const summary = asRecord(job.payload_summary);
  const resultSummary = asRecord(job.result_summary);
  const result = asRecord(job.result);
  const warnings = [
    ...asArray(job.warnings).map((item) => text(item, "")).filter(Boolean),
    ...asArray(result.payload_warnings).map((item) => text(item, "")).filter(Boolean),
  ];
  const audioUrl = text(resultSummary.audio_url || result.audio_url, "");
  const title = text(resultSummary.title || result.title || summary.title || summary.caption, "Song render");
  const artist = text(resultSummary.artist_name || result.artist_name || summary.artist_name, "");
  const gate = asRecord(result.vocal_intelligibility_gate || resultSummary.vocal_intelligibility_gate);
  const transcriptPreview = asArray(gate.transcript_preview);

  return (
    <div className="space-y-4">
      {audioUrl && (
        <WaveformPlayer
          src={audioUrl}
          title={title}
          artist={artist}
          metadata={{
            model: summary.song_model,
            quality: summary.quality_profile,
            duration: resultSummary.duration || summary.duration,
            bpm: resultSummary.bpm || summary.bpm,
            key: resultSummary.key_scale || summary.key_scale,
            seed: summary.seed,
            resultId: resultSummary.result_id,
          }}
        />
      )}

      <Section title="Render setup" icon={Music4}>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          <InfoRow label="Task" value={summary.task_type} />
          <InfoRow label="Model" value={summary.song_model || "auto"} />
          <InfoRow label="Quality" value={summary.quality_profile || "auto"} />
          <InfoRow label="Duration" value={summary.duration ? `${text(summary.duration)}s` : "auto"} />
          <InfoRow label="Instrumental" value={summary.instrumental} />
          <InfoRow label="Lyrics words" value={summary.lyrics_word_count} />
          <InfoRow label="Seed" value={summary.seed} />
          <InfoRow label="BPM / key" value={`${text(summary.bpm, "auto")} / ${text(summary.key_scale, "auto")}`} />
          <InfoRow label="Source audio" value={summary.has_source_audio} />
        </div>
      </Section>

      <Section title="Prompt">
        <div className="space-y-2 text-xs">
          <p className="rounded-md border bg-background/35 p-3 leading-relaxed">
            {text(summary.caption || result.caption, "Geen caption beschikbaar.")}
          </p>
          {Boolean(summary.tags) && (
            <p className="text-muted-foreground">
              Tags: {text(summary.tags)}
            </p>
          )}
        </div>
      </Section>

      {(warnings.length > 0 || Object.keys(gate).length > 0) && (
        <Section title="Warnings & vocal check" icon={AlertTriangle}>
          <div className="space-y-2 text-xs">
            {warnings.map((item, index) => (
              <p key={`${item}-${index}`} className="rounded-md bg-destructive/10 p-2 text-destructive">
                {item}
              </p>
            ))}
            {Object.keys(gate).length > 0 && (
              <div className="space-y-2">
                <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                  <InfoRow label="Vocal gate" value={gate.status} />
                  <InfoRow label="Passed" value={gate.passed} />
                  <InfoRow label="Attempt" value={`${text(gate.attempt)} / ${text(gate.max_attempts)}`} />
                  <InfoRow label="Needs review" value={gate.needs_review} />
                </div>
                {transcriptPreview.length > 0 && (
                  <div className="space-y-1">
                    {transcriptPreview.slice(0, 3).map((item, index) => {
                      const row = asRecord(item);
                      return (
                        <p key={index} className="rounded-md border bg-background/35 p-2 leading-relaxed">
                          {text(row.audio_id || `take ${index + 1}`)} · {text(row.status, "unknown")} —{" "}
                          {text(row.text || row.issue, "Geen transcript beschikbaar.")}
                        </p>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
          </div>
        </Section>
      )}

      <Section title="Log tail" icon={ScrollText}>
        <pre className="max-h-72 overflow-auto rounded-md bg-black/40 p-3 text-[11px] leading-relaxed text-muted-foreground">
          {log || asArray(job.logs).join("\n") || "Nog geen logregels beschikbaar."}
        </pre>
      </Section>
    </div>
  );
}

function GenericDetails({ job }: { job: JsonRecord }) {
  const payload = asRecord(job.payload);
  const result = asRecord(job.result);
  return (
    <div className="space-y-4">
      <Section title="Details">
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          <InfoRow label="Status" value={job.status || job.state} />
          <InfoRow label="Progress" value={`${text(job.progress, "0")}%`} />
          <InfoRow label="Updated" value={job.updated_at || job.finished_at || job.started_at} />
          <InfoRow label="Planner" value={job.planner_model || payload.planner_model} />
          <InfoRow label="Tracks" value={payload.num_tracks || result.track_count || asArray(result.tracks).length} />
          <InfoRow label="Error" value={job.error} />
        </div>
      </Section>
      <Section title="Raw snapshot">
        <pre className="max-h-80 overflow-auto rounded-md bg-black/40 p-3 text-[11px] leading-relaxed text-muted-foreground">
          {JSON.stringify(job, null, 2)}
        </pre>
      </Section>
    </div>
  );
}

function JobDetailsDialog({
  job,
  open,
  onOpenChange,
}: {
  job: JobEntry | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const navigate = useNavigate();
  const updateJob = useJobsStore((s) => s.updateJob);
  const [remoteJob, setRemoteJob] = React.useState<JsonRecord>({});
  const [log, setLog] = React.useState("");
  const [loading, setLoading] = React.useState(false);
  const [actionBusy, setActionBusy] = React.useState("");

  const load = React.useCallback(async () => {
    if (!job) return;
    setLoading(true);
    try {
      if (job.kind === "lora") {
        const [jobResp, logResp] = await Promise.all([
          api.get<{ success: boolean; job?: JsonRecord }>(job.detailsPath || `/api/lora/jobs/${encodeURIComponent(job.id)}`),
          api.get<{ success: boolean; log?: string }>(job.logPath || `/api/lora/jobs/${encodeURIComponent(job.id)}/log`),
        ]);
        const next = asRecord(jobResp.job);
        setRemoteJob(next);
        setLog(text(logResp.log, ""));
        updateJob(job.id, {
          progress: progressOf(job, next),
          status: stageOf(job, next) || stateOf(job, next),
          state: stateOf(job, next),
          stage: stageOf(job, next),
          updatedAt: next.updated_at ? text(next.updated_at) : Date.now(),
          metadata: next,
          error: text(next.error, ""),
        });
      } else if (job.kind === "generation") {
        const [jobResp, logResp] = await Promise.all([
          api.get<{ success: boolean; job?: JsonRecord }>(job.detailsPath || `/api/generation/jobs/${encodeURIComponent(job.id)}`),
          api.get<{ success: boolean; log?: string }>(job.logPath || `/api/generation/jobs/${encodeURIComponent(job.id)}/log`),
        ]);
        const next = asRecord(jobResp.job);
        setRemoteJob(next);
        setLog(text(logResp.log, ""));
        updateJob(job.id, {
          progress: progressOf(job, next),
          status: stageOf(job, next) || stateOf(job, next),
          state: stateOf(job, next),
          stage: stageOf(job, next),
          updatedAt: next.updated_at ? text(next.updated_at) : Date.now(),
          metadata: next,
          error: text(next.error, ""),
        });
      } else if (job.kind === "album") {
        const resp = await api.get<{ success: boolean; job?: JsonRecord }>(job.detailsPath || `/api/album/jobs/${encodeURIComponent(job.id)}`);
        const next = asRecord(resp.job);
        setRemoteJob(next);
        updateJob(job.id, {
          progress: progressOf(job, next),
          status: stageOf(job, next) || stateOf(job, next),
          state: stateOf(job, next),
          updatedAt: text(next.updated_at || next.finished_at, "") || Date.now(),
          metadata: next,
          error: text(next.error, ""),
        });
      } else if (job.kind === "ollama-pull") {
        const resp = await api.get<{ success: boolean; job?: JsonRecord }>(job.detailsPath || `/api/ollama/pull/${encodeURIComponent(job.id)}`);
        const next = asRecord(resp.job);
        setRemoteJob(next);
        updateJob(job.id, {
          progress: progressOf(job, next),
          status: text(next.status || next.state, job.status || ""),
          state: text(next.state || next.status, ""),
          updatedAt: text(next.updated_at, "") || Date.now(),
          metadata: next,
          error: text(next.error, ""),
        });
      } else {
        setRemoteJob(asRecord(job.metadata));
      }
    } catch (error) {
      toast.error(`Jobdetails laden mislukt: ${(error as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, [job, updateJob]);

  React.useEffect(() => {
    if (!open || !job) return;
    void load();
    const timer = window.setInterval(() => {
      const snapshot = asRecord(useJobsStore.getState().jobs[job.id]?.metadata);
      const state = stateOf(job, snapshot);
      if (!TERMINAL_STATES.has(state)) void load();
    }, 5000);
    return () => window.clearInterval(timer);
  }, [open, job?.id, load]);

  React.useEffect(() => {
    if (!open) {
      setRemoteJob({});
      setLog("");
    }
  }, [open]);

  if (!job) return null;

  const Icon = ICONS[job.kind] ?? Loader2;
  const remote = Object.keys(remoteJob).length ? remoteJob : asRecord(job.metadata);
  const state = stateOf(job, remote);
  const stage = stageOf(job, remote);
  const progress = progressOf(job, remote);
  const done = isDone(job, remote);
  const params = asRecord(remote.params);
  const canStop = job.kind === "lora" && ACTIVE_STATES.has(state);
  const canResume = job.kind === "lora" && ["failed", "stopped", "error"].includes(state);

  const runLoraAction = async (action: "stop" | "resume") => {
    setActionBusy(action);
    try {
      await api.post(`/api/lora/jobs/${encodeURIComponent(job.id)}/${action}`);
      toast.success(action === "stop" ? "Training wordt gestopt." : "Training wordt hervat.");
      await load();
    } catch (error) {
      toast.error((error as Error).message);
    } finally {
      setActionBusy("");
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[88vh] max-w-5xl overflow-hidden p-0">
        <DialogHeader className="border-b bg-card/80 p-5">
          <div className="flex items-start gap-3 pr-8">
            <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-primary/15 text-primary">
              {done ? <Icon className="size-5" /> : <Loader2 className="size-5 animate-spin" />}
            </div>
            <div className="min-w-0 flex-1">
              <DialogTitle className="truncate">{job.label}</DialogTitle>
              <DialogDescription>
                {job.kindLabel || kindLabel(job.kind)} · {job.id}
              </DialogDescription>
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <Badge variant={done ? "default" : "outline"}>{state}</Badge>
                {stage && <Badge variant="secondary">{stage}</Badge>}
                {Boolean(params.trigger_tag) && <Badge variant="muted">trigger {text(params.trigger_tag)}</Badge>}
              </div>
            </div>
            <div className="hidden text-right sm:block">
              <p className="font-mono text-lg">{Math.round(progress)}%</p>
              <p className="text-[10px] text-muted-foreground">progress</p>
            </div>
          </div>
          <Progress value={progress} className="mt-4 h-1.5" />
        </DialogHeader>

        <div className="max-h-[calc(88vh-150px)] space-y-4 overflow-auto p-5">
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" size="sm" onClick={() => void load()} disabled={loading}>
              {loading ? <Loader2 className="size-3.5 animate-spin" /> : <RefreshCw className="size-3.5" />}
              Refresh
            </Button>
            {canStop && (
              <Button variant="outline" size="sm" onClick={() => void runLoraAction("stop")} disabled={!!actionBusy}>
                {actionBusy === "stop" ? <Loader2 className="size-3.5 animate-spin" /> : <PauseCircle className="size-3.5" />}
                Stop
              </Button>
            )}
            {canResume && (
              <Button variant="outline" size="sm" onClick={() => void runLoraAction("resume")} disabled={!!actionBusy}>
                {actionBusy === "resume" ? <Loader2 className="size-3.5 animate-spin" /> : <PlayCircle className="size-3.5" />}
                Resume
              </Button>
            )}
            {job.kind === "lora" && done && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  onOpenChange(false);
                  navigate("/settings");
                }}
              >
                <ExternalLink className="size-3.5" />
                Open Settings
              </Button>
            )}
          </div>

          <Section title="Snapshot">
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
              <InfoRow label="Created" value={formatTime(remote.created_at || job.startedAt)} />
              <InfoRow label="Updated" value={formatTime(remote.updated_at || job.updatedAt)} />
              <InfoRow label="State" value={state} />
              <InfoRow label="Error" value={remote.error || job.error} />
            </div>
          </Section>

          {job.kind === "lora" ? (
            <LoraDetails job={remote} log={log} />
          ) : job.kind === "generation" ? (
            <GenerationDetails job={remote} log={log} />
          ) : (
            <GenericDetails job={remote} />
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

export function JobTracker({ compact = false }: { compact?: boolean }) {
  const jobs = useJobsStore((s) => s.jobs);
  const addJob = useJobsStore((s) => s.addJob);
  const removeJob = useJobsStore((s) => s.removeJob);
  const selectedId = useJobsStore((s) => s.selectedJobId);
  const openJob = useJobsStore((s) => s.openJob);
  const closeJob = useJobsStore((s) => s.closeJob);
  const list = Object.values(jobs).sort((a, b) => b.startedAt - a.startedAt);
  const selected = selectedId ? jobs[selectedId] ?? null : null;

  React.useEffect(() => {
    let cancelled = false;
    const hydrateRemoteJobs = async () => {
      try {
        const [loraResult, generationResult] = await Promise.allSettled([
          api.get<{ success: boolean; jobs?: JsonRecord[] }>("/api/lora/jobs"),
          api.get<{ success: boolean; jobs?: JsonRecord[] }>("/api/generation/jobs"),
        ]);
        if (cancelled) return;
        const current = useJobsStore.getState().jobs;
        const loraJobs = loraResult.status === "fulfilled" ? loraResult.value.jobs || [] : [];
        const generationJobs = generationResult.status === "fulfilled" ? generationResult.value.jobs || [] : [];
        for (const rawJob of loraJobs) {
          const id = text(rawJob.id, "");
          if (!id) continue;
          if (!isActiveJob(rawJob) && !current[id]) continue;
          addJob(jobPatchFromLora(rawJob));
        }
        for (const [index, rawJob] of generationJobs.entries()) {
          const id = text(rawJob.id || rawJob.task_id, "");
          if (!id) continue;
          if (!isActiveJob(rawJob) && !current[id] && index >= 8) continue;
          addJob(jobPatchFromGeneration(rawJob));
        }
      } catch {
        // The tracker should never make the whole app noisy when startup APIs are still warming.
      }
    };
    void hydrateRemoteJobs();
    const timer = window.setInterval(hydrateRemoteJobs, 7000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [addJob]);

  React.useEffect(() => {
    if (selectedId && !jobs[selectedId]) closeJob();
  }, [closeJob, jobs, selectedId]);

  if (list.length === 0) return null;

  return (
    <>
      <div className={cn("space-y-1.5 px-2 pb-2", compact && "rounded-lg border bg-card/95 p-2 shadow-lg backdrop-blur")}>
        <button
          className="flex w-full items-center justify-between gap-2 rounded-md px-1 py-1 text-left text-[10px] uppercase tracking-widest text-muted-foreground hover:bg-accent hover:text-accent-foreground"
          onClick={() => list[0]?.id && openJob(list[0].id)}
        >
          <span>Achtergrond-jobs</span>
          <Badge variant="outline">{list.length}</Badge>
        </button>
        <AnimatePresence initial={false}>
          {list.slice(0, compact ? 2 : 5).map((job) => {
            const Icon = ICONS[job.kind] ?? Loader2;
            const done = isDone(job, asRecord(job.metadata));
            const state = stateOf(job, asRecord(job.metadata));
            return (
              <motion.button
                type="button"
                key={job.id}
                layout
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, x: -8 }}
                transition={{ duration: 0.16 }}
                onClick={() => openJob(job.id)}
                className={cn(
                  "group flex w-full items-center gap-2 rounded-md border bg-card/50 px-2 py-1.5 text-left text-xs transition-colors hover:bg-accent/60",
                  state === "failed" || state === "error" ? "border-destructive/40" : "",
                  done && !(state === "failed" || state === "error") ? "border-primary/40" : "",
                )}
              >
                {done ? (
                  <CheckCircle2 className="size-3.5 shrink-0 text-primary" />
                ) : (
                  <Loader2 className="size-3.5 shrink-0 animate-spin text-primary" />
                )}
                <Icon className="size-3.5 shrink-0 text-muted-foreground" />
                <div className="min-w-0 flex-1 space-y-1">
                  <p className="truncate font-medium text-foreground">{job.label}</p>
                  {typeof job.progress === "number" && !done && (
                    <Progress value={job.progress} className="h-0.5" />
                  )}
                  <p className="truncate text-[10px] text-muted-foreground">{job.status || state}</p>
                </div>
                {done && (
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    className="size-5 opacity-0 transition-opacity group-hover:opacity-100"
                    onClick={(event) => {
                      event.stopPropagation();
                      removeJob(job.id);
                    }}
                    aria-label="Sluit"
                  >
                    <X className="size-3" />
                  </Button>
                )}
              </motion.button>
            );
          })}
        </AnimatePresence>
      </div>
      <JobDetailsDialog
        job={selected}
        open={Boolean(selected)}
        onOpenChange={(nextOpen) => {
          if (!nextOpen) closeJob();
        }}
      />
    </>
  );
}
