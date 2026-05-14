import * as React from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  Brain,
  CheckCircle2,
  Disc3,
  Download,
  ExternalLink,
  GraduationCap,
  Image as ImageIcon,
  ListMusic,
  Loader2,
  Music4,
  PauseCircle,
  PlayCircle,
  RefreshCw,
  ScrollText,
  Video,
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
import { GenerationAudioList, firstGenerationAudioUrl } from "@/components/wizard/GenerationAudioList";
import { api } from "@/lib/api";
import { audioBackendLabel } from "@/lib/audioBackend";
import { cn } from "@/lib/utils";
import { useJobsStore, type JobEntry, type JobKind } from "@/store/jobs";

const ICONS: Record<JobKind, React.ComponentType<{ className?: string }>> = {
  generation: Music4,
  "song-batch": ListMusic,
  album: Disc3,
  mflux: ImageIcon,
  "mlx-video": Video,
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

function firstRecord(value: unknown): JsonRecord {
  const items = asArray(value);
  return items.length > 0 ? asRecord(items[0]) : {};
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

function errorText(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  return String(error || "Onbekende fout");
}

function isNetworkFetchError(error: unknown): boolean {
  const message = errorText(error).toLowerCase();
  return (
    message.includes("failed to fetch") ||
    message.includes("networkerror") ||
    message.includes("load failed") ||
    message.includes("network request failed")
  );
}

function userFacingFetchError(error: unknown): string {
  if (isNetworkFetchError(error)) {
    return "Backend tijdelijk niet bereikbaar. Ik toon de laatste bekende jobdata en probeer opnieuw.";
  }
  return errorText(error);
}

function delay(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function getWithRetry<T>(path: string, attempts = 2): Promise<T> {
  let lastError: unknown;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      return await api.get<T>(path);
    } catch (error) {
      lastError = error;
      if (!isNetworkFetchError(error) || attempt === attempts - 1) break;
      await delay(350 + attempt * 450);
    }
  }
  throw lastError;
}

async function getOptionalLog(path: string): Promise<string | null> {
  try {
    const response = await getWithRetry<{ success: boolean; log?: string }>(path);
    return text(response.log, "");
  } catch {
    return null;
  }
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
  if (kind === "song-batch") return "Song batch";
  if (kind === "lora") return "LoRA training";
  if (kind === "mflux") return "MFLUX image";
  if (kind === "mlx-video") return "MLX video";
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

function jobPatchFromOllamaPull(job: JsonRecord): JobEntry {
  const id = text(job.id || job.model, "ollama-pull");
  const model = text(job.model, "Ollama model");
  return {
    id,
    kind: "ollama-pull",
    label: `pull ${model}`,
    progress: progressOf({ id, kind: "ollama-pull", label: "", startedAt: Date.now() }, job),
    status: text(job.status || job.state, "queued"),
    state: text(job.state || job.status, "queued"),
    kindLabel: "Ollama pull",
    detailsPath: `/api/ollama/pull/${encodeURIComponent(id)}`,
    metadata: job,
    error: text(job.error, ""),
    startedAt: job.started_at ? new Date(String(job.started_at)).getTime() : Date.now(),
    updatedAt: text(job.finished_at || job.started_at, ""),
  };
}

function jobPatchFromMflux(job: JsonRecord): JobEntry {
  const payload = asRecord(job.payload);
  const result = asRecord(job.result_summary || job.result);
  const model = asRecord(job.model);
  const id = text(job.id, "mflux-job");
  const label =
    text(payload.title || payload.prompt || result.image_url || model.label, "MFLUX image");
  return {
    id,
    kind: "mflux",
    label,
    progress: progressOf({ id, kind: "mflux", label: "", startedAt: Date.now() }, job),
    status: stageOf({ id, kind: "mflux", label: "", startedAt: Date.now() }, job) || stateOf({ id, kind: "mflux", label: "", startedAt: Date.now() }, job),
    state: text(job.state || job.status, "queued"),
    stage: text(job.stage || job.status, ""),
    kindLabel: "MFLUX image",
    detailsPath: `/api/mflux/jobs/${encodeURIComponent(id)}`,
    metadata: job,
    error: text(job.error, ""),
    startedAt: job.created_at ? new Date(String(job.created_at)).getTime() : Date.now(),
    updatedAt: text(job.updated_at || job.finished_at, ""),
  };
}

function jobPatchFromMlxVideo(job: JsonRecord): JobEntry {
  const payload = asRecord(job.payload);
  const result = asRecord(job.result_summary || job.result);
  const model = asRecord(job.model);
  const id = text(job.id, "mlx-video-job");
  const label = text(payload.title || payload.prompt || result.video_url || model.label, "MLX video");
  return {
    id,
    kind: "mlx-video",
    label,
    progress: progressOf({ id, kind: "mlx-video", label: "", startedAt: Date.now() }, job),
    status: stageOf({ id, kind: "mlx-video", label: "", startedAt: Date.now() }, job) || stateOf({ id, kind: "mlx-video", label: "", startedAt: Date.now() }, job),
    state: text(job.state || job.status, "queued"),
    stage: text(job.stage || job.status, ""),
    kindLabel: "MLX video",
    detailsPath: `/api/mlx-video/jobs/${encodeURIComponent(id)}`,
    metadata: job,
    error: text(job.error, ""),
    startedAt: job.created_at ? new Date(String(job.created_at)).getTime() : Date.now(),
    updatedAt: text(job.updated_at || job.finished_at, ""),
  };
}

function jobPatchFromAlbum(job: JsonRecord): JobEntry {
  const payload = asRecord(job.payload);
  const result = asRecord(job.result);
  const id = text(job.id, "album-job");
  const label = text(payload.album_title || result.album_title || payload.concept || id, "Album job");
  return {
    id,
    kind: "album",
    label,
    progress: progressOf({ id, kind: "album", label: "", startedAt: Date.now() }, job),
    status: stageOf({ id, kind: "album", label: "", startedAt: Date.now() }, job) || stateOf({ id, kind: "album", label: "", startedAt: Date.now() }, job),
    state: text(job.state || job.status, "queued"),
    stage: text(job.stage || job.status, ""),
    kindLabel: "Album job",
    detailsPath: `/api/album/jobs/${encodeURIComponent(id)}`,
    metadata: job,
    error: text(job.error, ""),
    startedAt: job.started_at ? new Date(String(job.started_at)).getTime() : Date.now(),
    updatedAt: text(job.updated_at || job.last_update_at || job.finished_at, ""),
  };
}

function jobPatchFromSongBatch(job: JsonRecord): JobEntry {
  const payload = asRecord(job.payload);
  const id = text(job.id, "song-batch-job");
  const label = text(job.batch_title || payload.batch_title || id, "Song batch");
  return {
    id,
    kind: "song-batch",
    label,
    progress: progressOf({ id, kind: "song-batch", label: "", startedAt: Date.now() }, job),
    status: stageOf({ id, kind: "song-batch", label: "", startedAt: Date.now() }, job) || stateOf({ id, kind: "song-batch", label: "", startedAt: Date.now() }, job),
    state: text(job.state || job.status, "queued"),
    stage: text(job.stage || job.status, ""),
    kindLabel: "Song batch",
    detailsPath: `/api/song-batches/jobs/${encodeURIComponent(id)}`,
    logPath: `/api/song-batches/jobs/${encodeURIComponent(id)}/log`,
    metadata: job,
    error: text(job.error || asArray(job.errors)[0], ""),
    startedAt: job.created_at ? new Date(String(job.created_at)).getTime() : Date.now(),
    updatedAt: text(job.updated_at || job.finished_at, ""),
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
  const lossHistory = asArray(result.loss_history).map(asRecord);
  const plateauStatus = asRecord(result.plateau_status);
  const targetEpochs = result.target_epochs || params.target_epochs || result.epochs || params.train_epochs || params.epochs;
  const completedEpochs = result.completed_epochs || (lossHistory.length > 0 ? lossHistory[lossHistory.length - 1]?.epoch : "");

  return (
    <div className="space-y-4">
      <Section title="Training setup" icon={GraduationCap}>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          <InfoRow label="Trigger" value={params.trigger_tag} />
          <InfoRow label="Dataset" value={params.dataset_id} />
          <InfoRow label="Samples" value={result.sample_count} />
          <InfoRow label="Epochs" value={targetEpochs} />
          <InfoRow label="Completed" value={completedEpochs ? `${text(completedEpochs)} / ${text(targetEpochs)}` : ""} />
          <InfoRow label="Stop policy" value={result.stop_policy || params.stop_policy} />
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

      <Section title="Loss & early stop">
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          <InfoRow label="Last loss" value={result.last_loss} />
          <InfoRow label="Best loss" value={result.best_loss} />
          <InfoRow label="Best epoch" value={result.best_loss_epoch} />
          <InfoRow label="Plateau status" value={plateauStatus.status || result.plateau_status} />
          <InfoRow label="Patience" value={result.loss_patience_epochs || params.loss_patience_epochs} />
          <InfoRow label="Early stop reason" value={result.early_stop_reason} />
        </div>
        {(Boolean(plateauStatus.reason) || Boolean(result.early_stop_message) || Boolean(result.loss_warning)) && (
          <p className="mt-2 rounded-md bg-background/40 p-2 text-xs text-muted-foreground">
            {text(result.early_stop_message || plateauStatus.reason || result.loss_warning)}
          </p>
        )}
        {lossHistory.length > 0 && (
          <div className="mt-3 grid gap-1 text-xs sm:grid-cols-2 lg:grid-cols-3">
            {lossHistory.slice(-6).map((item, index) => (
              <div key={index} className="rounded-md border bg-background/35 p-2">
                Epoch {text(item.epoch)} · loss {text(item.loss)} · {text(item.source)}
              </div>
            ))}
          </div>
        )}
      </Section>

      <Section title="Epoch audition">
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          <InfoRow label="Enabled" value={audition.enabled} />
          <InfoRow label="Style profile" value={audition.style_profile || audition.genre_profile || audition.genre} />
          <InfoRow label="Duration" value={`${text(audition.duration, "20")}s`} />
          <InfoRow label="Scale" value={audition.scale} />
          <InfoRow label="Lyrics source" value={audition.lyrics_source} />
          <InfoRow label="Caption" value={audition.caption} />
          <InfoRow label="Caption tags" value={audition.style_caption_tags} />
        </div>
        {auditions.length > 0 && (
          <div className="mt-3 space-y-2">
            {auditions.slice(-6).map((item, index) => (
              <div key={index} className="rounded-md border bg-background/35 p-2 text-xs">
                <div className="flex items-center justify-between gap-2">
                  <span>Epoch {text(item.epoch)}</span>
                  <Badge variant={text(item.status).toLowerCase() === "succeeded" ? "default" : "outline"}>
                    {text(item.status || item.state || item.success)}
                  </Badge>
                </div>
                <div className="mt-1 grid gap-1 sm:grid-cols-2">
                  <InfoRow label="Style" value={item.style_profile || item.genre_profile} />
                  <InfoRow
                    label="Style audit"
                    value={
                      typeof item.style_conditioning_audit === "object" && item.style_conditioning_audit
                        ? (item.style_conditioning_audit as Record<string, unknown>).status
                        : ""
                    }
                  />
                </div>
                {Boolean(item.audio_url) && (
                  <div className="mt-2 space-y-2">
                    <WaveformPlayer
                      src={text(item.audio_url)}
                      title={`Epoch ${text(item.epoch)} test-WAV`}
                      artist={text(params.trigger_tag || "LoRA audition")}
                      metadata={{
                        model: item.song_model || params.song_model,
                        duration: item.duration,
                        bpm: item.bpm,
                        key: item.keyscale,
                        resultId: item.result_id,
                      }}
                      className="bg-card/55"
                    />
                    <p className="truncate font-mono text-[10px] text-muted-foreground">
                      {text(item.audio_url)}
                    </p>
                  </div>
                )}
                {Boolean(item.transcript_preview) && (
                  <p className="mt-2 line-clamp-3 rounded-md bg-background/40 p-2 leading-relaxed text-muted-foreground">
                    {text(item.transcript_preview)}
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
  const payload = asRecord(job.payload);
  const resultSummary = asRecord(job.result_summary);
  const result = asRecord(job.result);
  const primaryAudio = firstRecord(result.audios);
  const audioLora = asRecord(primaryAudio.lora_adapter);
  const warnings = [
    ...asArray(job.warnings).map((item) => text(item, "")).filter(Boolean),
    ...asArray(result.payload_warnings).map((item) => text(item, "")).filter(Boolean),
  ];
  const audioUrl = firstGenerationAudioUrl(result) || text(resultSummary.audio_url || result.audio_url, "");
  const title = text(resultSummary.title || result.title || summary.title || summary.caption, "Song render");
  const artist = text(resultSummary.artist_name || result.artist_name || summary.artist_name, "");
  const gate = asRecord(result.vocal_intelligibility_gate || resultSummary.vocal_intelligibility_gate);
  const transcriptPreview = asArray(gate.transcript_preview);
  const vocalPreflight = asRecord(result.vocal_preflight || resultSummary.vocal_preflight);
  const loraPreflight = asRecord(result.lora_preflight || resultSummary.lora_preflight);
  const loraPreflightAttempts = asArray(loraPreflight.attempts);
  const vocalHistory = asArray(result.vocal_intelligibility_history || resultSummary.vocal_intelligibility_history);
  const diagnosticAttempts = asArray(result.diagnostic_attempts || resultSummary.diagnostic_attempts);
  const requestedModel = text(result.requested_song_model || resultSummary.requested_song_model || summary.song_model, "");
  const actualModel = text(result.actual_song_model || resultSummary.actual_song_model || summary.song_model, "");
  const memoryPolicy = asRecord(result.memory_policy || resultSummary.memory_policy || summary.memory_policy);
  const requestedTakeCount =
    result.requested_take_count ||
    resultSummary.requested_take_count ||
    summary.requested_take_count ||
    summary.batch_size;
  const actualRunnerBatchSize =
    result.actual_runner_batch_size ||
    resultSummary.actual_runner_batch_size ||
    summary.actual_runner_batch_size ||
    memoryPolicy.actual_runner_batch_size ||
    (memoryPolicy.force_runner_batch_size_one ? 1 : "");
  const memoryPolicyLabel =
    memoryPolicy.policy ||
    result.error_type ||
    (actualRunnerBatchSize ? "planned" : "");
  const loraActive =
    result.with_lora ??
    resultSummary.with_lora ??
    audioLora.use_lora ??
    summary.with_lora ??
    payload.use_lora;
  const loraScale =
    result.lora_scale ??
    resultSummary.lora_scale ??
    audioLora.scale ??
    summary.lora_scale ??
    payload.lora_scale;
  const loraAdapterName =
    result.lora_adapter_name ||
    resultSummary.lora_adapter_name ||
    audioLora.name ||
    summary.lora_adapter_name ||
    payload.lora_adapter_name ||
    shortPath(result.lora_adapter_path || resultSummary.lora_adapter_path || audioLora.path || summary.lora_adapter_path || payload.lora_adapter_path);
  const loraTriggerTag =
    result.lora_trigger_tag ||
    resultSummary.lora_trigger_tag ||
    summary.lora_trigger_tag ||
    payload.lora_trigger_tag;
  const loraTriggerAudit = asRecord(
    result.lora_trigger_conditioning_audit ||
      resultSummary.lora_trigger_conditioning_audit ||
      summary.lora_trigger_conditioning_audit ||
      payload.lora_trigger_conditioning_audit,
  );
  const loraTriggerSource =
    result.lora_trigger_source ||
    resultSummary.lora_trigger_source ||
    summary.lora_trigger_source ||
    payload.lora_trigger_source ||
    loraTriggerAudit.trigger_source ||
    (loraTriggerTag ? "metadata" : "");
  const requestedLoraScale = payload.lora_scale ?? summary.lora_scale;
  const audioBackendStatus = asRecord(
    result.audio_backend_status ||
      resultSummary.audio_backend_status ||
      summary.audio_backend_status ||
      payload.audio_backend_status,
  );
  const requestedAudioBackend =
    audioBackendStatus.requested_audio_backend ||
    result.requested_audio_backend ||
    resultSummary.requested_audio_backend ||
    payload.audio_backend ||
    summary.audio_backend;
  const effectiveAudioBackend =
    audioBackendStatus.effective_audio_backend ||
    result.effective_audio_backend ||
    resultSummary.effective_audio_backend ||
    result.audio_backend ||
    resultSummary.audio_backend ||
    summary.audio_backend;
  const effectiveMlxActive = audioBackendStatus.effective_mlx_dit_active;
  const audioBackendFallback = text(audioBackendStatus.fallback_reason, "");
  const audioBackend =
    effectiveAudioBackend ||
    payload.audio_backend ||
    (result.use_mlx_dit === true || resultSummary.use_mlx_dit === true || payload.use_mlx_dit === true ? "mlx" : "");
  const audioBackendDisplay =
    effectiveMlxActive === true
      ? "MLX active"
      : requestedAudioBackend === "mlx" && effectiveMlxActive === false
        ? `MLX requested; ${audioBackendFallback || "runner did not confirm MLX"}`
        : audioBackend
          ? audioBackendLabel(audioBackend)
          : "auto";
  const styleAudit = asRecord(
    result.style_conditioning_audit ||
      resultSummary.style_conditioning_audit ||
      summary.style_conditioning_audit ||
      payload.style_conditioning_audit,
  );
  const styleProfile =
    result.style_profile ||
    resultSummary.style_profile ||
    summary.style_profile ||
    payload.style_profile ||
    (summary.caption || summary.tags || payload.caption || payload.tags ? "manual caption/tags" : "auto");
  const styleAuditLabel =
    styleAudit.status ||
    (styleProfile === "manual caption/tags" ? "manual" : styleProfile ? "pending" : "");

  return (
    <div className="space-y-4">
      {audioUrl && (
        <GenerationAudioList
          result={Object.keys(result).length ? result : resultSummary}
          title={title}
          artist={artist}
        />
      )}

      <Section title="Render setup" icon={Music4}>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          <InfoRow label="Task" value={summary.task_type} />
          <InfoRow label="Model" value={summary.song_model || "auto"} />
          <InfoRow label="Requested model" value={requestedModel || "auto"} />
          <InfoRow label="Actual model" value={actualModel || "auto"} />
          <InfoRow label="Audio backend" value={audioBackendDisplay} />
          <InfoRow label="Requested backend" value={requestedAudioBackend ? audioBackendLabel(requestedAudioBackend) : "auto"} />
          <InfoRow label="Quality" value={summary.quality_profile || "auto"} />
          <InfoRow label="Duration" value={summary.duration ? `${text(summary.duration)}s` : "auto"} />
          <InfoRow label="Requested takes" value={requestedTakeCount} />
          <InfoRow label="Runner batch" value={actualRunnerBatchSize} />
          <InfoRow label="Memory policy" value={memoryPolicyLabel} />
          <InfoRow label="Instrumental" value={summary.instrumental} />
          <InfoRow label="Lyrics words" value={summary.lyrics_word_count} />
          <InfoRow label="Seed" value={summary.seed} />
          <InfoRow label="BPM / key" value={`${text(summary.bpm, "auto")} / ${text(summary.key_scale, "auto")}`} />
          <InfoRow label="Source audio" value={summary.has_source_audio} />
          <InfoRow label="Attempt role" value={result.attempt_role || resultSummary.attempt_role || "primary"} />
          <InfoRow label="LoRA active" value={loraActive} />
          <InfoRow label="LoRA adapter" value={loraAdapterName} />
          <InfoRow label="LoRA trigger" value={loraTriggerTag} />
          <InfoRow label="LoRA trigger source" value={loraTriggerSource} />
          <InfoRow label="LoRA scale" value={loraScale} />
          <InfoRow label="Requested LoRA scale" value={requestedLoraScale} />
          <InfoRow label="Style profile" value={styleProfile} />
          <InfoRow label="Style audit" value={styleAuditLabel} />
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

      {(Object.keys(vocalPreflight).length > 0 || Object.keys(loraPreflight).length > 0 || vocalHistory.length > 0 || diagnosticAttempts.length > 0) && (
        <Section title="Attempts" icon={Activity}>
          <div className="space-y-2 text-xs">
            {Object.keys(vocalPreflight).length > 0 && (
              <div className="rounded-md border bg-background/35 p-3">
                <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                  <InfoRow label="Vocal preflight" value={vocalPreflight.status} />
                  <InfoRow label="Duration" value={vocalPreflight.duration ? `${text(vocalPreflight.duration)}s` : ""} />
                  <InfoRow label="Required" value={vocalPreflight.required_for_long_render} />
                </div>
                {Object.keys(asRecord(vocalPreflight.attempt)).length > 0 && (
                  <p className="mt-2 rounded-md bg-muted/35 p-2">
                    Primary preflight · model {text(asRecord(vocalPreflight.attempt).actual_song_model)} ·{" "}
                    {text(asRecord(vocalPreflight.attempt).vocal_gate_status)} · result{" "}
                    {text(asRecord(vocalPreflight.attempt).result_id)}
                  </p>
                )}
              </div>
            )}
            {Object.keys(loraPreflight).length > 0 && (
              <div className="rounded-md border bg-background/35 p-3">
                <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                  <InfoRow label="LoRA preflight" value={loraPreflight.status} />
                  <InfoRow label="Selected scale" value={loraPreflight.selected_scale} />
                  <InfoRow label="Adapter" value={shortPath(loraPreflight.adapter_path)} />
                </div>
                {loraPreflightAttempts.length > 0 && (
                  <div className="mt-2 space-y-1">
                    {loraPreflightAttempts.map((item, index) => {
                      const row = asRecord(item);
                      return (
                        <p key={index} className="rounded-md bg-muted/35 p-2">
                          {text(row.label || `attempt ${index + 1}`)} · LoRA {text(row.use_lora)} · scale{" "}
                          {text(row.lora_scale)} · {text(row.gate_status)} · result {text(row.result_id)}
                        </p>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
            {vocalHistory.length > 0 && (
              <div className="space-y-1">
                {vocalHistory.map((item, index) => {
                  const row = asRecord(item);
                  return (
                    <p key={index} className="rounded-md border bg-background/35 p-2">
                      Attempt {text(row.attempt || index + 1)} · {text(row.attempt_role || "primary")} · requested{" "}
                      {text(row.requested_song_model || requestedModel)} · actual {text(row.actual_song_model || row.song_model)} · LoRA{" "}
                      {text(row.with_lora)} · scale {text(row.lora_scale)} · result {text(row.result_id)} · {text(row.status)}
                    </p>
                  );
                })}
              </div>
            )}
            {diagnosticAttempts.length > 0 && (
              <div className="rounded-md border border-amber-500/30 bg-amber-500/10 p-3">
                <p className="mb-2 font-medium text-amber-700 dark:text-amber-300">
                  Diagnostic fallback only. Deze pogingen vervangen de primaire render niet.
                </p>
                <div className="space-y-1">
                  {diagnosticAttempts.map((item, index) => {
                    const row = asRecord(item);
                    return (
                      <p key={index} className="rounded-md bg-background/45 p-2">
                        {text(row.label || `diagnostic ${index + 1}`)} · actual {text(row.actual_song_model)} · LoRA{" "}
                        {text(row.with_lora)} · {text(row.vocal_gate_status)} · passed {text(row.passed)} · result{" "}
                        {text(row.result_id)} {row.failure_reason ? `· ${text(row.failure_reason)}` : ""}
                      </p>
                    );
                  })}
                </div>
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

function MfluxDetails({ job }: { job: JsonRecord }) {
  const payload = asRecord(job.payload);
  const result = asRecord(job.result || job.result_summary);
  const model = asRecord(job.model);
  const imageUrl = text(result.image_url || result.url, "");
  const logs = asArray(job.logs).map((item) => text(item, "")).filter(Boolean);
  return (
    <div className="space-y-4">
      {imageUrl && (
        <div className="overflow-hidden rounded-lg border bg-background/40">
          <img src={imageUrl} alt="MFLUX result" className="max-h-[520px] w-full object-contain" />
        </div>
      )}
      <Section title="Image setup" icon={ImageIcon}>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          <InfoRow label="Action" value={payload.action || result.action} />
          <InfoRow label="Model" value={model.label || result.model_label || payload.model_id} />
          <InfoRow label="Preset" value={model.preset} />
          <InfoRow label="Size" value={`${text(payload.width || result.width, "auto")} × ${text(payload.height || result.height, "auto")}`} />
          <InfoRow label="Steps" value={payload.steps || result.steps} />
          <InfoRow label="Seed" value={payload.seed || result.seed} />
          <InfoRow label="Quantize" value={payload.quantize || result.quantize} />
          <InfoRow label="LoRAs" value={asArray(payload.lora_adapters || result.lora_adapters).length} />
        </div>
      </Section>
      <Section title="Prompt">
        <p className="rounded-md border bg-background/35 p-3 text-xs leading-relaxed">
          {text(payload.prompt || result.prompt, "Geen prompt beschikbaar.")}
        </p>
      </Section>
      <Section title="Log tail" icon={ScrollText}>
        <pre className="max-h-72 overflow-auto rounded-md bg-black/40 p-3 text-[11px] leading-relaxed text-muted-foreground">
          {logs.join("\n") || "Nog geen logregels beschikbaar."}
        </pre>
      </Section>
    </div>
  );
}

function MlxVideoDetails({ job }: { job: JsonRecord }) {
  const payload = asRecord(job.payload);
  const result = asRecord(job.result || job.result_summary);
  const model = asRecord(job.model);
  const videoUrl = text(result.video_url || result.url, "");
  const posterUrl = text(result.poster_url, "");
  const logs = asArray(job.logs || result.logs).map((item) => text(item, "")).filter(Boolean);
  return (
    <div className="space-y-4">
      {videoUrl && (
        <div className="overflow-hidden rounded-lg border bg-black/70">
          <video
            src={videoUrl}
            poster={posterUrl || undefined}
            controls
            className="max-h-[560px] w-full bg-black object-contain"
          />
        </div>
      )}
      <Section title="Video setup" icon={Video}>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          <InfoRow label="Action" value={payload.action || result.action} />
          <InfoRow label="Model" value={model.label || result.model_label || payload.model_id} />
          <InfoRow label="Engine" value={model.engine || result.engine} />
          <InfoRow label="Preset" value={model.preset || result.preset} />
          <InfoRow label="Size" value={`${text(payload.width || result.width, "auto")} × ${text(payload.height || result.height, "auto")}`} />
          <InfoRow label="Frames/FPS" value={`${text(payload.num_frames || result.num_frames, "auto")} / ${text(payload.fps || result.fps, "auto")}`} />
          <InfoRow label="Steps" value={payload.steps || result.steps} />
          <InfoRow label="Seed" value={payload.seed || result.seed} />
          <InfoRow label="LoRAs" value={asArray(payload.lora_adapters || result.lora_adapters).length} />
        </div>
      </Section>
      <Section title="Prompt">
        <p className="rounded-md border bg-background/35 p-3 text-xs leading-relaxed">
          {text(payload.prompt || result.prompt, "Geen prompt beschikbaar.")}
        </p>
      </Section>
      <Section title="Sources">
        <div className="grid gap-2 md:grid-cols-2">
          <InfoRow label="Image" value={payload.image_path || payload.source_image_path || result.source_image} />
          <InfoRow label="Audio" value={payload.audio_path || payload.source_audio_path || result.source_audio} />
        </div>
      </Section>
      <Section title="Log tail" icon={ScrollText}>
        <pre className="max-h-72 overflow-auto rounded-md bg-black/40 p-3 text-[11px] leading-relaxed text-muted-foreground">
          {logs.join("\n") || "Nog geen logregels beschikbaar."}
        </pre>
      </Section>
    </div>
  );
}

function albumPlayableTracks(result: JsonRecord): JsonRecord[] {
  const seen = new Set<string>();
  const rows: JsonRecord[] = [];

  const push = (raw: unknown) => {
    const track = asRecord(raw);
    const audios = asArray(track.audios).map(asRecord);
    if (!audios.some((audio) => text(audio.audio_url || audio.download_url || audio.library_url, ""))) return;
    const firstAudio = audios[0] || {};
    const key = [
      track.album_id || firstAudio.album_id,
      track.result_id || firstAudio.result_id,
      track.track_number,
      track.album_model || track.active_song_model || firstAudio.album_model || firstAudio.song_model,
      track.title || firstAudio.title,
    ].map((item) => text(item, "")).join("::");
    if (seen.has(key)) return;
    seen.add(key);
    rows.push(track);
  };

  asArray(result.tracks).forEach((rawTrack) => {
    const track = asRecord(rawTrack);
    const modelResults = asArray(track.model_results);
    if (modelResults.length) {
      modelResults.forEach(push);
    } else {
      push(track);
    }
  });

  if (!rows.length) {
    asArray(result.model_albums).forEach((rawAlbum) => {
      const album = asRecord(rawAlbum);
      asArray(album.tracks).forEach((rawTrack) => {
        const track = { ...asRecord(rawTrack) };
        if (!track.album_id) track.album_id = album.album_id;
        if (!track.album_model) track.album_model = album.album_model;
        if (!track.album_model_label) track.album_model_label = album.album_model_label;
        push(track);
      });
    });
  }

  return rows;
}

function AlbumDetails({ job }: { job: JsonRecord }) {
  const payload = asRecord(job.payload);
  const result = asRecord(job.result);
  const playableTracks = albumPlayableTracks(result);
  const logs = asArray(job.logs || result.logs).map((item) => text(item, "")).filter(Boolean);
  const familyDownload = text(result.family_download_url || job.download_url, "");
  const albumDownload = text(result.download_url, "");
  return (
    <div className="space-y-4">
      <Section title="Album setup" icon={Disc3}>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          <InfoRow label="Status" value={job.status || result.album_status || job.state} />
          <InfoRow label="Progress" value={`${text(job.progress, "0")}%`} />
          <InfoRow label="Stage" value={job.stage} />
          <InfoRow label="Current task" value={job.current_task} />
          <InfoRow label="Current agent" value={job.current_agent} />
          <InfoRow
            label="Track progress"
            value={`${text(job.current_track, "0")} / ${text(job.total_tracks || payload.num_tracks || result.track_count, "?")}`}
          />
          <InfoRow label="Completed tracks" value={job.completed_tracks || result.completed_track_count} />
          <InfoRow label="Remaining tracks" value={job.remaining_tracks} />
          <InfoRow
            label="LLM wait"
            value={Boolean(job.waiting_on_llm) ? `${text(job.llm_provider || "LLM")} ${text(job.llm_wait_elapsed_s, "0")}s` : "no"}
          />
          <InfoRow label="Last update" value={job.last_update_at || job.updated_at} />
          <InfoRow label="Tracks" value={payload.num_tracks || result.track_count || asArray(result.tracks).length} />
          <InfoRow label="Full tracks ready" value={result.full_tracks_ready || result.completed_track_count || playableTracks.length} />
          <InfoRow label="Audio files ready" value={result.completed_audio_count || asArray(result.audios).length} />
          <InfoRow label="Expected renders" value={job.expected_count || result.expected_renders} />
          <InfoRow label="Album family" value={result.album_family_id || job.album_family_id} />
          <InfoRow label="Error" value={job.error || result.error} />
        </div>
        {(familyDownload || albumDownload) && (
          <div className="mt-3 flex flex-wrap gap-2">
            {familyDownload && (
              <Button asChild variant="outline" size="sm">
                <a href={familyDownload} download>
                  <Download className="size-3.5" />
                  Download family ZIP
                </a>
              </Button>
            )}
            {albumDownload && (
              <Button asChild variant="outline" size="sm">
                <a href={albumDownload} download>
                  <Download className="size-3.5" />
                  Download album ZIP
                </a>
              </Button>
            )}
          </div>
        )}
      </Section>

      <Section title="Volledige tracks klaar" icon={Music4}>
        {playableTracks.length ? (
          <div className="space-y-4">
            <p className="text-xs text-muted-foreground">
              Elke kaart hieronder is een volledig gerenderde track/take. Nieuwe tracks verschijnen hier zodra ze klaar zijn.
            </p>
            {playableTracks.map((track, index) => (
              <div key={`${text(track.result_id, "track")}-${index}`} className="rounded-lg border bg-background/35 p-3">
                <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
                  <Badge variant="secondary">Track {text(track.track_number || index + 1)}</Badge>
                  {Boolean(track.album_model_label || track.album_model || track.active_song_model) && (
                    <Badge variant="outline">{text(track.album_model_label || track.album_model || track.active_song_model)}</Badge>
                  )}
                  {Boolean(track.result_id) && <Badge variant="muted">{text(track.result_id)}</Badge>}
                </div>
                <GenerationAudioList
                  result={track}
                  title={text(track.title, `Track ${index + 1}`)}
                  artist={text(track.artist_name, "")}
                />
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">
            Nog geen volledige track klaar. Zodra de eerste track klaar is, verschijnt hier direct een speler terwijl de rest doorrendert.
          </p>
        )}
      </Section>

      <Section title="Log tail" icon={ScrollText}>
        <pre className="max-h-72 overflow-auto rounded-md bg-black/40 p-3 text-[11px] leading-relaxed text-muted-foreground">
          {logs.join("\n") || "Nog geen logregels beschikbaar."}
        </pre>
      </Section>
    </div>
  );
}

function SongBatchDetails({
  job,
  log,
}: {
  job: JsonRecord;
  log: string;
}) {
  const payload = asRecord(job.payload);
  const songs = asArray(job.songs).map(asRecord);
  const logs = log ? log.split("\n") : asArray(job.logs).map((item) => text(item, "")).filter(Boolean);
  const activeSong = songs.find((song) => ["running", "queued"].includes(String(song.state || "").toLowerCase()));
  return (
    <div className="space-y-4">
      <Section title="Batch setup" icon={ListMusic}>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          <InfoRow label="Batch title" value={job.batch_title || payload.batch_title} />
          <InfoRow label="Current song" value={job.current_song ? `${text(job.current_song)} / ${text(job.total_songs)}` : ""} />
          <InfoRow label="Completed" value={job.completed_songs} />
          <InfoRow label="Failed" value={job.failed_songs} />
          <InfoRow label="Remaining" value={job.remaining_songs} />
          <InfoRow label="Child job" value={job.child_generation_job_id} />
          <InfoRow label="Stop on error" value={payload.stop_on_error} />
          <InfoRow label="Active title" value={activeSong?.title} />
        </div>
      </Section>

      <Section title="Songs">
        <div className="space-y-3">
          {songs.map((song, index) => {
            const result = asRecord(song.result);
            const summary = asRecord(song.payload_summary);
            const songProgress = Number(song.progress);
            return (
              <div key={`${song.track_number || index}-${song.generation_job_id || song.title}`} className="rounded-lg border bg-background/35 p-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium">
                      {text(song.track_number || index + 1)}. {text(song.title, `Song ${index + 1}`)}
                    </p>
                    <p className="truncate text-xs text-muted-foreground">
                      {text(summary.song_model || asRecord(song.payload).song_model, "auto")} · job {text(song.generation_job_id)}
                    </p>
                  </div>
                  <Badge variant={String(song.state).toLowerCase() === "failed" ? "destructive" : String(song.state).toLowerCase() === "succeeded" ? "default" : "outline"}>
                    {text(song.status || song.state)}
                  </Badge>
                </div>
                {Number.isFinite(songProgress) && (
                  <Progress value={Math.max(0, Math.min(100, songProgress))} className="mt-2 h-1" />
                )}
                {Boolean(song.error) && (
                  <p className="mt-2 rounded-md bg-destructive/10 p-2 text-xs text-destructive">
                    {text(song.error)}
                  </p>
                )}
                {Object.keys(result).length > 0 && (
                  <GenerationAudioList
                    result={result}
                    title={text(song.title, `Song ${index + 1}`)}
                    artist={text(summary.artist_name, "")}
                    className="mt-3 space-y-2"
                  />
                )}
              </div>
            );
          })}
        </div>
      </Section>

      <Section title="Log tail" icon={ScrollText}>
        <pre className="max-h-72 overflow-auto rounded-md bg-black/40 p-3 text-[11px] leading-relaxed text-muted-foreground">
          {logs.join("\n") || "Nog geen logregels beschikbaar."}
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
  const [detailError, setDetailError] = React.useState("");
  const [actionBusy, setActionBusy] = React.useState("");
  const loadingRef = React.useRef(false);
  const failureCountRef = React.useRef(0);
  const lastToastAtRef = React.useRef(0);

  const load = React.useCallback(async (options: { quiet?: boolean } = {}) => {
    if (!job || loadingRef.current) return;
    loadingRef.current = true;
    setLoading(true);
    try {
      if (job.kind === "lora") {
        const jobResp = await getWithRetry<{ success: boolean; job?: JsonRecord }>(
          job.detailsPath || `/api/lora/jobs/${encodeURIComponent(job.id)}`,
        );
        const nextLog = await getOptionalLog(job.logPath || `/api/lora/jobs/${encodeURIComponent(job.id)}/log`);
        const next = asRecord(jobResp.job);
        setRemoteJob(next);
        if (nextLog !== null) setLog(nextLog);
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
        const jobResp = await getWithRetry<{ success: boolean; job?: JsonRecord }>(
          job.detailsPath || `/api/generation/jobs/${encodeURIComponent(job.id)}`,
        );
        const nextLog = await getOptionalLog(job.logPath || `/api/generation/jobs/${encodeURIComponent(job.id)}/log`);
        const next = asRecord(jobResp.job);
        setRemoteJob(next);
        if (nextLog !== null) setLog(nextLog);
        updateJob(job.id, {
          progress: progressOf(job, next),
          status: stageOf(job, next) || stateOf(job, next),
          state: stateOf(job, next),
          stage: stageOf(job, next),
          updatedAt: next.updated_at ? text(next.updated_at) : Date.now(),
          metadata: next,
          error: text(next.error, ""),
        });
      } else if (job.kind === "song-batch") {
        const jobResp = await getWithRetry<{ success: boolean; job?: JsonRecord }>(
          job.detailsPath || `/api/song-batches/jobs/${encodeURIComponent(job.id)}`,
        );
        const nextLog = await getOptionalLog(job.logPath || `/api/song-batches/jobs/${encodeURIComponent(job.id)}/log`);
        const next = asRecord(jobResp.job);
        setRemoteJob(next);
        if (nextLog !== null) setLog(nextLog);
        updateJob(job.id, {
          progress: progressOf(job, next),
          status: stageOf(job, next) || stateOf(job, next),
          state: stateOf(job, next),
          stage: stageOf(job, next),
          updatedAt: next.updated_at ? text(next.updated_at) : Date.now(),
          metadata: next,
          error: text(next.error || asArray(next.errors)[0], ""),
        });
      } else if (job.kind === "album") {
        const resp = await getWithRetry<{ success: boolean; job?: JsonRecord }>(
          job.detailsPath || `/api/album/jobs/${encodeURIComponent(job.id)}`,
        );
        const next = asRecord(resp.job);
        setRemoteJob(next);
        updateJob(job.id, {
          progress: progressOf(job, next),
          status: stageOf(job, next) || stateOf(job, next),
          state: stateOf(job, next),
          stage: stageOf(job, next),
          updatedAt: text(next.updated_at || next.finished_at, "") || Date.now(),
          metadata: next,
          error: text(next.error, ""),
        });
      } else if (job.kind === "mflux") {
        const resp = await getWithRetry<{ success: boolean; job?: JsonRecord }>(
          job.detailsPath || `/api/mflux/jobs/${encodeURIComponent(job.id)}`,
        );
        const next = asRecord(resp.job);
        setRemoteJob(next);
        updateJob(job.id, {
          progress: progressOf(job, next),
          status: stageOf(job, next) || stateOf(job, next),
          state: stateOf(job, next),
          stage: stageOf(job, next),
          updatedAt: text(next.updated_at || next.finished_at, "") || Date.now(),
          metadata: next,
          error: text(next.error, ""),
        });
      } else if (job.kind === "mlx-video") {
        const resp = await getWithRetry<{ success: boolean; job?: JsonRecord }>(
          job.detailsPath || `/api/mlx-video/jobs/${encodeURIComponent(job.id)}`,
        );
        const next = asRecord(resp.job);
        setRemoteJob(next);
        updateJob(job.id, {
          progress: progressOf(job, next),
          status: stageOf(job, next) || stateOf(job, next),
          state: stateOf(job, next),
          stage: stageOf(job, next),
          updatedAt: text(next.updated_at || next.finished_at, "") || Date.now(),
          metadata: next,
          error: text(next.error, ""),
        });
      } else if (job.kind === "ollama-pull") {
        const resp = await getWithRetry<{ success: boolean; job?: JsonRecord }>(
          job.detailsPath || `/api/ollama/pull/${encodeURIComponent(job.id)}`,
        );
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
      failureCountRef.current = 0;
      setDetailError("");
    } catch (error) {
      failureCountRef.current += 1;
      const message = userFacingFetchError(error);
      setDetailError(message);
      const shouldToast =
        !options.quiet &&
        Date.now() - lastToastAtRef.current > 8000;
      if (shouldToast) {
        lastToastAtRef.current = Date.now();
        toast.error(`Jobdetails laden mislukt: ${message}`);
      }
    } finally {
      setLoading(false);
      loadingRef.current = false;
    }
  }, [job, updateJob]);

  React.useEffect(() => {
    if (!open || !job) return;
    void load({ quiet: true });
    const timer = window.setInterval(() => {
      const currentJob = useJobsStore.getState().jobs[job.id] || job;
      const snapshot = asRecord(currentJob.metadata);
      const state = stateOf(currentJob, snapshot);
      if (!TERMINAL_STATES.has(state)) void load({ quiet: true });
    }, 5000);
    return () => window.clearInterval(timer);
  }, [open, job?.id, load]);

  React.useEffect(() => {
    if (!open) {
      setRemoteJob({});
      setLog("");
      setDetailError("");
      failureCountRef.current = 0;
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
  const canStopAlbum = job.kind === "album" && ["queued", "running"].includes(state);

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

  const stopAlbumJob = async () => {
    setActionBusy("album-stop");
    try {
      await api.post(`/api/album/jobs/${encodeURIComponent(job.id)}/stop`);
      toast.success("Album-job stopverzoek gestuurd.");
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
            <Button variant="outline" size="sm" onClick={() => void load({ quiet: false })} disabled={loading}>
              {loading ? <Loader2 className="size-3.5 animate-spin" /> : <RefreshCw className="size-3.5" />}
              Refresh
            </Button>
            {canStop && (
              <Button variant="outline" size="sm" onClick={() => void runLoraAction("stop")} disabled={!!actionBusy}>
                {actionBusy === "stop" ? <Loader2 className="size-3.5 animate-spin" /> : <PauseCircle className="size-3.5" />}
                Stop
              </Button>
            )}
            {canStopAlbum && (
              <Button variant="outline" size="sm" onClick={() => void stopAlbumJob()} disabled={!!actionBusy}>
                {actionBusy === "album-stop" ? <Loader2 className="size-3.5 animate-spin" /> : <PauseCircle className="size-3.5" />}
                Stop album
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

          {detailError && (
            <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-900 dark:text-amber-100">
              <div className="flex items-start gap-2">
                <AlertTriangle className="mt-0.5 size-4 shrink-0" />
                <div>
                  <p className="font-medium">Jobdetails tijdelijk niet ververst</p>
                  <p className="mt-1 text-muted-foreground">{detailError}</p>
                </div>
              </div>
            </div>
          )}

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
          ) : job.kind === "song-batch" ? (
            <SongBatchDetails job={remote} log={log} />
          ) : job.kind === "album" ? (
            <AlbumDetails job={remote} />
          ) : job.kind === "mflux" ? (
            <MfluxDetails job={remote} />
          ) : job.kind === "mlx-video" ? (
            <MlxVideoDetails job={remote} />
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
        const [loraResult, generationResult, songBatchResult, mfluxResult, mlxVideoResult, albumResult, ollamaResult] = await Promise.allSettled([
          api.get<{ success: boolean; jobs?: JsonRecord[] }>("/api/lora/jobs"),
          api.get<{ success: boolean; jobs?: JsonRecord[] }>("/api/generation/jobs"),
          api.get<{ success: boolean; jobs?: JsonRecord[] }>("/api/song-batches/jobs"),
          api.get<{ success: boolean; jobs?: JsonRecord[] }>("/api/mflux/jobs"),
          api.get<{ success: boolean; jobs?: JsonRecord[] }>("/api/mlx-video/jobs"),
          api.get<{ success: boolean; jobs?: JsonRecord[] }>("/api/album/jobs"),
          api.get<{ success: boolean; pull_jobs?: JsonRecord[] }>("/api/ollama/status"),
        ]);
        if (cancelled) return;
        const current = useJobsStore.getState().jobs;
        const loraJobs = loraResult.status === "fulfilled" ? loraResult.value.jobs || [] : [];
        const generationJobs = generationResult.status === "fulfilled" ? generationResult.value.jobs || [] : [];
        const songBatchJobs = songBatchResult.status === "fulfilled" ? songBatchResult.value.jobs || [] : [];
        const mfluxJobs = mfluxResult.status === "fulfilled" ? mfluxResult.value.jobs || [] : [];
        const mlxVideoJobs = mlxVideoResult.status === "fulfilled" ? mlxVideoResult.value.jobs || [] : [];
        const albumJobs = albumResult.status === "fulfilled" ? albumResult.value.jobs || [] : [];
        const ollamaPullJobs = ollamaResult.status === "fulfilled" ? ollamaResult.value.pull_jobs || [] : [];
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
        for (const [index, rawJob] of songBatchJobs.entries()) {
          const id = text(rawJob.id, "");
          if (!id) continue;
          if (!isActiveJob(rawJob) && !current[id] && index >= 8) continue;
          addJob(jobPatchFromSongBatch(rawJob));
        }
        for (const [index, rawJob] of mfluxJobs.entries()) {
          const id = text(rawJob.id, "");
          if (!id) continue;
          if (!isActiveJob(rawJob) && !current[id] && index >= 8) continue;
          addJob(jobPatchFromMflux(rawJob));
        }
        for (const [index, rawJob] of mlxVideoJobs.entries()) {
          const id = text(rawJob.id, "");
          if (!id) continue;
          if (!isActiveJob(rawJob) && !current[id] && index >= 8) continue;
          addJob(jobPatchFromMlxVideo(rawJob));
        }
        for (const [index, rawJob] of albumJobs.entries()) {
          const id = text(rawJob.id, "");
          if (!id) continue;
          if (!isActiveJob(rawJob) && !current[id] && index >= 8) continue;
          addJob(jobPatchFromAlbum(rawJob));
        }
        for (const rawJob of ollamaPullJobs) {
          const id = text(rawJob.id || rawJob.model, "");
          if (!id) continue;
          if (!isActiveJob(rawJob) && !current[id]) continue;
          addJob(jobPatchFromOllamaPull(rawJob));
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
