import * as React from "react";
import { useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
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
  api,
  getLoraAutolabelJob,
  startLoraAutolabelJob,
  type LoraAutolabelJob,
} from "@/lib/api";
import { useJobsStore } from "@/store/jobs";
import { toast } from "@/components/ui/sonner";
import { cn } from "@/lib/utils";

interface TrainerForm {
  dataset_id: string;
  trigger_tag: string;
  default_language: string;
  default_bpm: number;
  default_time_signature: string;
  learning_rate: number;
  train_epochs: number;
  batch_size: number;
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
  progress?: number;
  step?: number;
  total_steps?: number;
}

const ALLOWED_AUDIO = /\.(wav|mp3|flac|ogg|m4a|aac)$/i;

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
    default_language: "en",
    default_bpm: 120,
    default_time_signature: "4/4",
    learning_rate: 1e-4,
    train_epochs: 10,
    batch_size: 1,
  });

  const addJob = useJobsStore((s) => s.addJob);
  const updateJobStore = useJobsStore((s) => s.updateJob);
  const removeJob = useJobsStore((s) => s.removeJob);

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
      setDataset({
        dataset_id: resp.dataset_id,
        files: resp.copied_files?.length ?? 0,
        copied_files: resp.copied_files,
        skipped_files: resp.skipped_files,
        status: "imported",
      });
      setForm((f) => ({ ...f, dataset_id: resp.dataset_id! }));
      toast.success(
        `${resp.copied_files?.length ?? 0} audio-bestanden geïmporteerd & gelabeld.`,
      );
    },
    onError: (e: Error) => toast.error(e.message),
  });

  // ---- AI auto-label (understand_music) job ------------------------------

  const startAutolabel = useMutation({
    mutationFn: () =>
      startLoraAutolabelJob({
        dataset_id: dataset!.dataset_id,
        language: form.default_language,
        skip_existing: true,
      }),
    onSuccess: (resp) => {
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
      addJob({
        id,
        kind: "lora",
        label: `Auto-label ${dataset!.dataset_id.slice(0, 12)}`,
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
            state?: string;
            status?: string;
            progress?: number;
            step?: number;
            total_steps?: number;
            error?: string;
          };
        }>(`/api/lora/jobs/${encodeURIComponent(job.id)}`);
        if (cancelled) return;
        const j = resp.job;
        if (!j) return;
        const state = (j.state ?? "running").toLowerCase();
        const description = j.status ?? state;
        const p = typeof j.progress === "number" ? j.progress : 0;
        setJob({
          id: job.id,
          state,
          status: description,
          progress: p,
          step: j.step,
          total_steps: j.total_steps,
        });
        updateJobStore(job.id, { progress: p, status: description });
        if (state === "complete" || state === "succeeded") {
          toast.success("LoRA training klaar.");
          updateJobStore(job.id, { status: "complete", progress: 100 });
          setTimeout(() => removeJob(job.id), 4000);
          return;
        }
        if (state === "error" || state === "failed") {
          toast.error(j.error ?? "Training mislukt");
          updateJobStore(job.id, { status: "error" });
          setTimeout(() => removeJob(job.id), 6000);
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
        </div>
      ),
    },
    {
      key: "autolabel",
      title: "AI transcribe & label",
      description:
        "Laat ACE-Step's understand_music elke clip beluisteren en automatisch lyrics + caption + bpm/key afleiden. Schrijft .lyrics.txt en .json sidecar-bestanden zodat training echte lyric-conditioning krijgt. Per ACE-Step docs: review de getranscribeerde lyrics achteraf op fouten.",
      isValid:
        !!dataset?.dataset_id &&
        (autolabelSkipped ||
          (autolabelJob?.state ?? "").toLowerCase() === "complete"),
      hidden: !dataset?.dataset_id,
      render: () => (
        <div className="space-y-4">
          {!autolabelJob && !autolabelSkipped && (
            <FieldGroup
              title="Heb je vocal-tracks of pure instrumentals?"
              description="Bij vocals: laat AI lyrics extraheren. Bij instrumentals: skip — alle samples krijgen dan [Instrumental]."
            >
              <div className="grid gap-3 sm:grid-cols-2">
                <Button
                  onClick={() => startAutolabel.mutate()}
                  disabled={startAutolabel.isPending}
                  className="gap-2"
                >
                  {startAutolabel.isPending ? (
                    <Loader2 className="size-4 animate-spin" />
                  ) : (
                    <Mic2 className="size-4" />
                  )}
                  Start AI auto-label
                </Button>
                <Button
                  variant="outline"
                  onClick={() => setAutolabelSkipped(true)}
                  className="gap-2"
                >
                  <SkipForward className="size-4" />
                  Skip (alles instrumental)
                </Button>
              </div>
            </FieldGroup>
          )}

          {autolabelJob && (
            <div className="space-y-3 rounded-xl border border-primary/30 bg-primary/5 p-4 text-sm">
              <div className="flex items-center gap-3">
                <Loader2
                  className={cn(
                    "size-5 text-primary",
                    (autolabelJob.state ?? "") !== "complete" &&
                      (autolabelJob.state ?? "") !== "error" &&
                      "animate-spin",
                  )}
                />
                <div className="min-w-0 flex-1">
                  <p className="truncate font-medium">
                    {autolabelJob.status ?? autolabelJob.state ?? "—"}
                  </p>
                  <p className="truncate text-[10px] text-muted-foreground">
                    {autolabelJob.current_file || `${autolabelJob.processed ?? 0}/${autolabelJob.total ?? 0}`}
                  </p>
                </div>
                <span className="font-mono text-sm tabular-nums">
                  {autolabelJob.progress ?? 0}%
                </span>
              </div>
              <Progress value={autolabelJob.progress ?? 0} />
              <div className="flex flex-wrap gap-2 text-[11px]">
                {typeof autolabelJob.succeeded === "number" && (
                  <Badge variant="muted">{autolabelJob.succeeded} succeeded</Badge>
                )}
                {typeof autolabelJob.failed === "number" && autolabelJob.failed > 0 && (
                  <Badge variant="destructive">{autolabelJob.failed} failed</Badge>
                )}
                {typeof autolabelJob.total === "number" && (
                  <Badge variant="muted">{autolabelJob.total} totaal</Badge>
                )}
              </div>
              {autolabelJob.errors && autolabelJob.errors.length > 0 && (
                <details className="rounded-md border bg-background/40 p-2 text-xs">
                  <summary className="cursor-pointer font-medium">
                    {autolabelJob.errors.length} fout(en)
                  </summary>
                  <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap text-[10px] text-destructive">
                    {autolabelJob.errors.join("\n")}
                  </pre>
                </details>
              )}
              {autolabelJob.logs && autolabelJob.logs.length > 0 && (
                <details className="rounded-md border bg-background/40 p-2 text-xs">
                  <summary className="cursor-pointer font-medium">
                    {autolabelJob.logs.length} log entries
                  </summary>
                  <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap text-[10px] text-muted-foreground">
                    {autolabelJob.logs.join("\n")}
                  </pre>
                </details>
              )}
            </div>
          )}

          {autolabelSkipped && (
            <div className="rounded-md border border-yellow-500/30 bg-yellow-500/5 p-3 text-xs text-yellow-200">
              Auto-label overgeslagen. Alle samples worden als
              <code className="mx-1 rounded bg-background/40 px-1">[Instrumental]</code>
              getraind. Vocal/lyric-epoch auditions worden niet gevalideerd.
            </div>
          )}
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
        <div className="rounded-xl border border-primary/30 bg-primary/5 p-5 text-sm">
          <div className="flex items-center gap-3">
            <Loader2
              className={
                job?.state === "complete" || job?.state === "error"
                  ? "size-5 text-primary"
                  : "size-5 animate-spin text-primary"
              }
            />
            <div className="flex-1">
              <p className="font-medium">{job?.status ?? job?.state ?? "—"}</p>
              <p className="text-xs text-muted-foreground">
                {job?.step != null && job?.total_steps != null
                  ? `step ${job.step}/${job.total_steps}`
                  : `job ${job?.id ?? ""}`}
              </p>
            </div>
            <span className="font-mono text-sm">{job?.progress ?? 0}%</span>
          </div>
          <Progress value={job?.progress ?? 0} className="mt-3" />
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
