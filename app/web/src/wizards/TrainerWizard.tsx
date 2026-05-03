import * as React from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { Loader2, Music4, Upload, GraduationCap, FolderOpen } from "lucide-react";

import { WizardShell, FieldGroup, type WizardStepDef } from "@/components/wizard/WizardShell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { toast } from "@/components/ui/sonner";

interface TrainerForm {
  dataset_name: string;
  trigger_tag: string;
  default_language: string;
  default_bpm: number;
  default_time_signature: string;
  learning_rate: number;
  train_epochs: number;
  batch_size: number;
}

interface DatasetState {
  dataset_id?: string;
  files?: number;
  status?: string;
}

interface TrainJobState {
  job_id?: string;
  status?: string;
  progress?: number;
  step?: number;
  total_steps?: number;
}

export function TrainerWizard() {
  const navigate = useNavigate();
  const [step, setStep] = React.useState(0);
  const [folderPath, setFolderPath] = React.useState("");
  const [dataset, setDataset] = React.useState<DatasetState | null>(null);
  const [job, setJob] = React.useState<TrainJobState | null>(null);
  const [form, setForm] = React.useState<TrainerForm>({
    dataset_name: "",
    trigger_tag: "",
    default_language: "en",
    default_bpm: 120,
    default_time_signature: "4/4",
    learning_rate: 1e-4,
    train_epochs: 10,
    batch_size: 1,
  });

  const importFolder = useMutation({
    mutationFn: () =>
      api.post<{ success: boolean; dataset_id?: string; files?: number; error?: string }>(
        "/api/lora/dataset/import-folder",
        {
          path: folderPath,
          dataset_name: form.dataset_name || undefined,
        },
      ),
    onSuccess: (resp) => {
      if (!resp.success || !resp.dataset_id) {
        toast.error(resp.error || "Import mislukt");
        return;
      }
      setDataset({
        dataset_id: resp.dataset_id,
        files: resp.files,
        status: "imported",
      });
      toast.success(`Dataset met ${resp.files ?? "?"} bestanden geïmporteerd.`);
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const autolabel = useMutation({
    mutationFn: () =>
      api.post<{ success: boolean; error?: string }>(
        "/api/lora/dataset/autolabel",
        {
          dataset_id: dataset?.dataset_id,
          default_language: form.default_language,
          default_bpm: form.default_bpm,
          default_time_signature: form.default_time_signature,
          trigger_tag: form.trigger_tag,
        },
      ),
    onSuccess: (resp) => {
      if (!resp.success) {
        toast.error(resp.error || "Autolabel mislukt");
        return;
      }
      setDataset((d) => (d ? { ...d, status: "labeled" } : d));
      toast.success("Autolabel klaar.");
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const startTrain = useMutation({
    mutationFn: () =>
      api.post<{ success: boolean; job_id?: string; error?: string }>(
        "/api/lora/one-click-train",
        {
          dataset_id: dataset?.dataset_id,
          trigger_tag: form.trigger_tag,
          training_defaults: {
            learning_rate: form.learning_rate,
            train_epochs: form.train_epochs,
            batch_size: form.batch_size,
          },
        },
      ),
    onSuccess: (resp) => {
      if (!resp.success || !resp.job_id) {
        toast.error(resp.error || "Training kon niet starten");
        return;
      }
      setJob({ job_id: resp.job_id, status: "queued", progress: 0 });
      setStep(steps.length - 1);
    },
    onError: (e: Error) => toast.error(e.message),
  });

  // poll training job
  React.useEffect(() => {
    if (!job?.job_id) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const resp = await api.get<{
          success: boolean;
          job?: { status?: string; progress?: number; step?: number; total_steps?: number; error?: string };
        }>(`/api/lora/train/${encodeURIComponent(job.job_id!)}`).catch(() => null);
        if (!resp || cancelled) return;
        const j = resp.job;
        if (!j) return;
        setJob((prev) => prev && {
          ...prev,
          status: j.status,
          progress: j.progress ?? prev.progress,
          step: j.step,
          total_steps: j.total_steps,
        });
        if (j.status === "complete") {
          toast.success("LoRA training klaar.");
          return;
        }
        if (j.status === "error") {
          toast.error(j.error || "Training mislukt");
          return;
        }
        setTimeout(poll, 4000);
      } catch (e) {
        toast.error(`Poll fout: ${(e as Error).message}`);
      }
    };
    poll();
    return () => {
      cancelled = true;
    };
  }, [job?.job_id]);

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
            <li>Wijs een trigger-tag aan (bv. <code className="rounded bg-background/40 px-1">char_aurora</code>) zodat je hem later kunt activeren.</li>
            <li>Verzamel ~20–100 audio-bestanden in één map met dezelfde stijl/stem.</li>
            <li>De AI labelt elke clip; je hoeft dat niet handmatig te doen.</li>
            <li>Training duurt ~10–60 min, afhankelijk van GPU en epoch-aantal.</li>
          </ul>
        </div>
      ),
    },
    {
      key: "dataset",
      title: "Dataset importeren",
      description: "Wijs een map aan met audio-bestanden (wav/mp3/flac).",
      isValid: !!dataset?.dataset_id,
      render: () => (
        <div className="space-y-4">
          <FieldGroup title="Dataset">
            <div className="space-y-1.5">
              <Label>Naam</Label>
              <Input
                value={form.dataset_name}
                onChange={(e) => setForm({ ...form, dataset_name: e.target.value })}
                placeholder="bv. aurora-songbook"
              />
            </div>
            <div className="space-y-1.5">
              <Label className="flex items-center gap-1.5">
                <FolderOpen className="size-3.5" /> Pad naar de audio-map
              </Label>
              <Input
                value={folderPath}
                onChange={(e) => setFolderPath(e.target.value)}
                placeholder="/Users/.../mijn-tracks"
              />
            </div>
            <Button
              onClick={() => importFolder.mutate()}
              disabled={!folderPath || importFolder.isPending}
              className="w-full gap-2"
            >
              <Upload className="size-4" />
              {importFolder.isPending ? "Importeren…" : "Importeer map"}
            </Button>
            {dataset?.dataset_id && (
              <motion.div
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-center gap-2 text-xs text-muted-foreground"
              >
                <Badge variant="muted">id: {dataset.dataset_id.slice(0, 12)}…</Badge>
                <Badge variant="muted">{dataset.files ?? "?"} bestanden</Badge>
                <Badge variant="muted">status: {dataset.status}</Badge>
              </motion.div>
            )}
          </FieldGroup>
        </div>
      ),
    },
    {
      key: "label",
      title: "Auto-label",
      description: "AI verzint per audio-bestand een caption, tags, lyrics-fragment en metadata.",
      isValid: dataset?.status === "labeled",
      render: () => (
        <div className="space-y-4">
          <FieldGroup title="Defaults">
            <div className="grid gap-3 sm:grid-cols-3">
              <div className="space-y-1.5">
                <Label>Trigger tag</Label>
                <Input
                  value={form.trigger_tag}
                  onChange={(e) => setForm({ ...form, trigger_tag: e.target.value })}
                  placeholder="char_aurora"
                />
              </div>
              <div className="space-y-1.5">
                <Label>Default taal</Label>
                <Input
                  value={form.default_language}
                  onChange={(e) => setForm({ ...form, default_language: e.target.value })}
                />
              </div>
              <div className="space-y-1.5">
                <Label>Default bpm</Label>
                <Input
                  type="number"
                  value={form.default_bpm}
                  onChange={(e) => setForm({ ...form, default_bpm: Number(e.target.value) })}
                />
              </div>
            </div>
            <Button
              onClick={() => autolabel.mutate()}
              disabled={!dataset?.dataset_id || !form.trigger_tag || autolabel.isPending}
              className="w-full gap-2"
            >
              {autolabel.isPending ? <Loader2 className="size-4 animate-spin" /> : <GraduationCap className="size-4" />}
              {autolabel.isPending ? "Labelen…" : "Start auto-label"}
            </Button>
          </FieldGroup>
        </div>
      ),
    },
    {
      key: "train",
      title: "Training-instellingen",
      isValid: true,
      render: () => (
        <FieldGroup title="Hyperparameters">
          <div className="grid gap-4 sm:grid-cols-3">
            <div className="space-y-3">
              <div className="flex items-baseline justify-between">
                <Label>Learning rate</Label>
                <span className="font-mono text-xs">{form.learning_rate.toExponential(1)}</span>
              </div>
              <Slider
                value={[Math.log10(form.learning_rate * 1e6)]}
                min={0}
                max={4}
                step={0.5}
                onValueChange={(v) => setForm({ ...form, learning_rate: Math.pow(10, (v[0] ?? 2) - 6) })}
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
                onValueChange={(v) => setForm({ ...form, train_epochs: v[0] ?? 10 })}
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
                onValueChange={(v) => setForm({ ...form, batch_size: v[0] ?? 1 })}
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
      isValid: !job || job.status === "complete",
      hidden: !job,
      render: () => (
        <div className="rounded-xl border border-primary/30 bg-primary/5 p-5 text-sm">
          <div className="flex items-center gap-3">
            <Loader2
              className={
                job?.status === "running"
                  ? "size-5 animate-spin text-primary"
                  : "size-5 text-primary"
              }
            />
            <div className="flex-1">
              <p className="font-medium">{job?.status ?? "—"}</p>
              <p className="text-xs text-muted-foreground">
                {job?.step != null && job?.total_steps != null
                  ? `step ${job.step}/${job.total_steps}`
                  : "—"}
              </p>
            </div>
            <span className="font-mono text-sm">{job?.progress ?? 0}%</span>
          </div>
          <Progress value={job?.progress ?? 0} className="mt-3" />
          <div className="mt-4">
            <Button variant="outline" onClick={() => navigate("/settings")} className="gap-2">
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
      isFinishing={startTrain.isPending || (job ? job.status !== "complete" && job.status !== "error" : false)}
      finishLabel={
        job ? "Training loopt…" : startTrain.isPending ? "Job start…" : "Start training"
      }
    />
  );
}
