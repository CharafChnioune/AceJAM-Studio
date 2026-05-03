import * as React from "react";
import { toast } from "@/components/ui/sonner";
import {
  getGenerationJob,
  startGenerationJob,
  type GenerateAdvancedResponse,
  type GenerationJob,
  type WizardMode,
} from "@/lib/api";
import { useJobsStore, type JobEntry } from "@/store/jobs";

const DONE_STATES = new Set(["succeeded", "complete", "completed", "success", "failed", "error", "stopped"]);

function text(value: unknown, fallback = ""): string {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function progressOf(job?: GenerationJob): number {
  const raw = Number(job?.progress ?? 0);
  return Number.isFinite(raw) ? Math.max(0, Math.min(100, raw)) : 0;
}

function jobEntryFromGeneration(job: GenerationJob, fallbackLabel: string): JobEntry {
  const summary = job.payload_summary ?? {};
  const result = job.result_summary ?? {};
  const id = text(job.id || job.task_id, "generation-job");
  const label = text(result.title || summary.title || summary.caption || fallbackLabel, fallbackLabel);
  return {
    id,
    kind: "generation",
    label,
    progress: progressOf(job),
    status: text(job.stage || job.status || job.state, "Queued"),
    state: text(job.state, "queued"),
    stage: text(job.stage || job.status, ""),
    kindLabel: "Song render",
    detailsPath: `/api/generation/jobs/${encodeURIComponent(id)}`,
    logPath: `/api/generation/jobs/${encodeURIComponent(id)}/log`,
    updatedAt: job.updated_at || Date.now(),
    metadata: job as unknown as Record<string, unknown>,
    error: text(job.error, ""),
    startedAt: job.created_at ? new Date(job.created_at).getTime() : Date.now(),
  };
}

interface UseGenerationJobRunnerOptions {
  mode: WizardMode;
  label: string;
  onComplete: (result: GenerateAdvancedResponse) => void;
}

export function useGenerationJobRunner({
  mode,
  label,
  onComplete,
}: UseGenerationJobRunnerOptions) {
  const addJob = useJobsStore((s) => s.addJob);
  const updateJob = useJobsStore((s) => s.updateJob);
  const openJob = useJobsStore((s) => s.openJob);
  const [jobId, setJobId] = React.useState<string>("");
  const [activeJob, setActiveJob] = React.useState<GenerationJob | null>(null);
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const completedRef = React.useRef("");
  const timerRef = React.useRef<number | null>(null);

  const clearTimer = React.useCallback(() => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const syncJob = React.useCallback(
    (job: GenerationJob) => {
      const entry = jobEntryFromGeneration(job, label);
      setActiveJob(job);
      addJob(entry);
      updateJob(entry.id, {
        progress: entry.progress,
        status: entry.status,
        state: entry.state,
        stage: entry.stage,
        updatedAt: entry.updatedAt,
        metadata: entry.metadata,
        error: entry.error,
      });
      return entry.id;
    },
    [addJob, label, updateJob],
  );

  const poll = React.useCallback(
    async (id: string) => {
      const resp = await getGenerationJob(id);
      if (!resp.success || !resp.job) {
        throw new Error(resp.error || "Generation job niet gevonden");
      }
      const nextId = syncJob(resp.job);
      const state = text(resp.job.state, "").toLowerCase();
      if (!DONE_STATES.has(state)) return;
      clearTimer();
      if (completedRef.current === nextId) return;
      completedRef.current = nextId;
      if (state === "succeeded" && resp.job.result?.success !== false) {
        const result = resp.job.result as GenerateAdvancedResponse;
        onComplete(result);
        toast.success(`"${result.title || label}" is klaar.`);
      } else {
        toast.error(resp.job.error || resp.job.result?.error || "Generatie mislukte");
      }
    },
    [clearTimer, label, onComplete, syncJob],
  );

  const start = React.useCallback(
    async (payload: Record<string, unknown>) => {
      clearTimer();
      completedRef.current = "";
      setIsSubmitting(true);
      try {
        const resp = await startGenerationJob({ ...payload, wizard_mode: mode });
        if (!resp.success || !resp.job_id || !resp.job) {
          throw new Error(resp.error || "Generatiejob starten mislukte");
        }
        const id = syncJob(resp.job);
        setJobId(id);
        toast.success("Render draait op de achtergrond.");
        timerRef.current = window.setInterval(() => {
          void poll(id).catch((error) => toast.error((error as Error).message));
        }, 2500);
        void poll(id).catch((error) => toast.error((error as Error).message));
      } catch (error) {
        toast.error((error as Error).message);
      } finally {
        setIsSubmitting(false);
      }
    },
    [clearTimer, mode, poll, syncJob],
  );

  React.useEffect(() => clearTimer, [clearTimer]);

  return {
    activeJob,
    jobId,
    isSubmitting,
    isRunning: Boolean(jobId && activeJob && !DONE_STATES.has(text(activeJob.state).toLowerCase())),
    openActiveJob: () => jobId && openJob(jobId),
    start,
  };
}
