import * as React from "react";
import { useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import {
  Loader2, Music4, Upload, GraduationCap, X, FileMusic,
} from "lucide-react";

import { WizardShell, FieldGroup, type WizardStepDef } from "@/components/wizard/WizardShell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";
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
        state: "queued",
        stage: "queued",
        kindLabel: "LoRA training",
        detailsPath: `/api/lora/jobs/${encodeURIComponent(id)}`,
        logPath: `/api/lora/jobs/${encodeURIComponent(id)}/log`,
        metadata: {
          dataset_id: dataset?.dataset_id,
          trigger_tag: form.trigger_tag,
          language: form.default_language,
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
        setJob({
          id: job.id,
          state,
          status: description,
          progress: p,
          step: j.step,
          total_steps: j.total_steps,
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
