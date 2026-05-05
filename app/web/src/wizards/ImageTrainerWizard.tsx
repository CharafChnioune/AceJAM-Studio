import * as React from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { GraduationCap, Loader2 } from "lucide-react";

import { WizardShell, FieldGroup, type WizardStepDef } from "@/components/wizard/WizardShell";
import { ReviewStep } from "@/components/wizard/ReviewStep";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { toast } from "@/components/ui/sonner";
import {
  getMfluxJob,
  getMfluxLoras,
  getMfluxModels,
  startMfluxLoraTraining,
  type MfluxJob,
} from "@/lib/api";
import { useJobsStore } from "@/store/jobs";

export function ImageTrainerWizard() {
  const addJob = useJobsStore((s) => s.addJob);
  const updateJob = useJobsStore((s) => s.updateJob);
  const [step, setStep] = React.useState(0);
  const [triggerTag, setTriggerTag] = React.useState("");
  const [datasetPath, setDatasetPath] = React.useState("");
  const [datasetType, setDatasetType] = React.useState<"txt2img" | "edit">("txt2img");
  const [modelId, setModelId] = React.useState("flux2-klein-9b");
  const [epochs, setEpochs] = React.useState(1);
  const [batchSize, setBatchSize] = React.useState(1);
  const [learningRate, setLearningRate] = React.useState("0.0001");
  const [seed, setSeed] = React.useState("42");
  const [quantize, setQuantize] = React.useState(8);
  const [saveFrequency, setSaveFrequency] = React.useState(30);
  const [generateImageFrequency, setGenerateImageFrequency] = React.useState(30);
  const [previewPrompt, setPreviewPrompt] = React.useState("");
  const [jobId, setJobId] = React.useState("");
  const [job, setJob] = React.useState<MfluxJob | null>(null);

  const modelsQ = useQuery({ queryKey: ["mflux", "models"], queryFn: getMfluxModels, staleTime: 60_000 });
  const adaptersQ = useQuery({ queryKey: ["mflux", "loras"], queryFn: getMfluxLoras, staleTime: 20_000 });
  const models = (modelsQ.data?.models ?? []).filter((model) => model.trainable);
  const status = modelsQ.data?.status;

  const payload = {
    trigger_tag: triggerTag,
    display_name: triggerTag,
    dataset_path: datasetPath,
    dataset_type: datasetType,
    model_id: modelId,
    epochs,
    batch_size: batchSize,
    learning_rate: Number(learningRate) || 1e-4,
    seed: Number(seed) || 42,
    quantize,
    save_frequency: saveFrequency,
    generate_image_frequency: generateImageFrequency,
    preview_prompt: previewPrompt || `${triggerTag || "style-token"}, premium album artwork, no text`,
  };

  React.useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    const poll = async () => {
      const resp = await getMfluxJob(jobId);
      if (cancelled || !resp.job) return;
      setJob(resp.job);
      updateJob(jobId, {
        progress: resp.job.progress,
        status: resp.job.status || resp.job.state,
        state: resp.job.state,
        stage: resp.job.stage,
        metadata: resp.job as unknown as Record<string, unknown>,
        error: resp.job.error,
      });
      if (!["succeeded", "failed", "error"].includes(String(resp.job.state))) {
        window.setTimeout(poll, 2500);
      }
    };
    void poll();
    return () => {
      cancelled = true;
    };
  }, [jobId, updateJob]);

  const train = useMutation({
    mutationFn: () => startMfluxLoraTraining(payload),
    onSuccess: (resp) => {
      if (!resp.success || !resp.job?.id) {
        toast.error(resp.error || "Image-LoRA training kon niet starten");
        return;
      }
      setJobId(resp.job.id);
      setJob(resp.job);
      addJob({
        id: resp.job.id,
        kind: "mflux",
        label: `image LoRA ${triggerTag || resp.job.id}`,
        progress: resp.job.progress || 0,
        status: resp.job.status || "queued",
        state: resp.job.state || "queued",
        stage: resp.job.stage,
        kindLabel: "MFLUX image LoRA",
        detailsPath: `/api/mflux/jobs/${encodeURIComponent(resp.job.id)}`,
        metadata: resp.job as unknown as Record<string, unknown>,
        startedAt: Date.now(),
      });
      toast.success("MFLUX image-LoRA training gestart.");
      setStep(3);
    },
    onError: (error: Error) => toast.error(error.message),
  });

  const steps: WizardStepDef[] = [
    {
      key: "dataset",
      title: "Dataset",
      description: "Map met afbeeldingen en captions/config voor MFLUX training.",
      isValid: datasetPath.trim().length > 0 && triggerTag.trim().length > 0,
      render: () => (
        <FieldGroup title="Image dataset">
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label>Trigger tag</Label>
              <Input value={triggerTag} onChange={(e) => setTriggerTag(e.target.value)} placeholder="album-cover-style" />
            </div>
            <div className="space-y-1.5">
              <Label>Dataset path</Label>
              <Input value={datasetPath} onChange={(e) => setDatasetPath(e.target.value)} placeholder="/Users/.../image-dataset" />
            </div>
          </div>
          <div className="mt-3 grid gap-3 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label>Dataset layout</Label>
              <Select value={datasetType} onValueChange={(value) => setDatasetType(value as "txt2img" | "edit")}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="txt2img">txt2img · image + same-name .txt captions</SelectItem>
                  <SelectItem value="edit">edit · _in / _out image pairs</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label>Preview prompt</Label>
              <Input
                value={previewPrompt}
                onChange={(e) => setPreviewPrompt(e.target.value)}
                placeholder={`${triggerTag || "trigger"}, premium album artwork`}
              />
            </div>
          </div>
          <div className="mt-4 rounded-md border bg-background/35 p-3 text-xs text-muted-foreground">
            <p className="font-medium text-foreground">Expected MFLUX dataset</p>
            <p>txt2img: each image should have a matching <code>.txt</code> caption. Edit: use paired files like <code>cover_001_in.png</code> and <code>cover_001_out.png</code>.</p>
          </div>
        </FieldGroup>
      ),
    },
    {
      key: "training",
      title: "Training",
      description: "Trainbare presets volgen de MFLUX catalogus.",
      isValid: status?.ready !== false,
      render: () => (
        <div className="space-y-4">
          {status && !status.ready && (
            <p className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
              {status.blocking_reason || "MFLUX is niet klaar."}
            </p>
          )}
          <FieldGroup title="Hyperparameters">
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1.5">
                <Label>Base model</Label>
                <Select value={modelId} onValueChange={setModelId}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {models.map((model) => (
                      <SelectItem key={model.id} value={model.id}>{model.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label>Learning rate</Label>
                <Input value={learningRate} onChange={(e) => setLearningRate(e.target.value)} />
              </div>
              <div className="space-y-1.5">
                <Label>Epochs</Label>
                <Input type="number" value={epochs} onChange={(e) => setEpochs(Number(e.target.value) || 1)} />
              </div>
              <div className="space-y-1.5">
                <Label>Batch size</Label>
                <Input type="number" value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value) || 1)} />
              </div>
              <div className="space-y-1.5">
                <Label>Seed</Label>
                <Input value={seed} onChange={(e) => setSeed(e.target.value)} />
              </div>
              <div className="space-y-1.5">
                <Label>Quantization</Label>
                <Input type="number" min={4} max={8} value={quantize} onChange={(e) => setQuantize(Number(e.target.value) || 8)} />
              </div>
              <div className="space-y-1.5">
                <Label>Save every</Label>
                <Input type="number" value={saveFrequency} onChange={(e) => setSaveFrequency(Number(e.target.value) || 30)} />
              </div>
              <div className="space-y-1.5">
                <Label>Preview every</Label>
                <Input type="number" value={generateImageFrequency} onChange={(e) => setGenerateImageFrequency(Number(e.target.value) || 30)} />
              </div>
            </div>
          </FieldGroup>
        </div>
      ),
    },
    {
      key: "review",
      title: "Review",
      isValid: true,
      render: () => (
        <ReviewStep
          payload={payload}
          warnings={[]}
          primaryFields={[
            { key: "trigger_tag", label: "Trigger" },
            { key: "model_id", label: "Model" },
            { key: "dataset_type", label: "Layout" },
            { key: "epochs", label: "Epochs" },
            { key: "batch_size", label: "Batch" },
            { key: "learning_rate", label: "LR" },
            { key: "preview_prompt", label: "Preview" },
            { key: "dataset_path", label: "Dataset" },
          ]}
        />
      ),
    },
    {
      key: "result",
      title: "Job",
      isValid: true,
      render: () => (
        <div className="space-y-4">
          <div className="rounded-xl border bg-card/45 p-5">
            <div className="flex items-center gap-3">
              {job && !["succeeded", "failed"].includes(String(job.state)) ? (
                <Loader2 className="size-5 animate-spin text-primary" />
              ) : (
                <GraduationCap className="size-5 text-primary" />
              )}
              <div>
                <p className="font-medium">{job?.stage || job?.status || "Klaar om te trainen"}</p>
                {job && <p className="text-xs text-muted-foreground">{job.id}</p>}
              </div>
              {job && <Badge variant="outline" className="ml-auto">{job.progress ?? 0}%</Badge>}
            </div>
          </div>
          <Button onClick={() => train.mutate()} disabled={train.isPending || status?.ready === false}>
            {train.isPending ? <Loader2 className="size-4 animate-spin" /> : <GraduationCap className="size-4" />}
            Start training
          </Button>
          {job?.error && <p className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">{job.error}</p>}
          {job?.dataset_summary && (
            <FieldGroup title="Dataset summary">
              <pre className="max-h-56 overflow-auto rounded-md bg-muted p-3 text-xs">
                {JSON.stringify(job.dataset_summary, null, 2)}
              </pre>
            </FieldGroup>
          )}
          <FieldGroup title="Known image LoRAs">
            {(adaptersQ.data?.adapters ?? []).length === 0 ? (
              <p className="text-sm text-muted-foreground">Nog geen image-LoRAs geregistreerd.</p>
            ) : (
              <div className="space-y-2">
                {(adaptersQ.data?.adapters ?? []).map((adapter) => (
                  <div key={adapter.path} className="rounded-md border bg-background/35 p-2 text-sm">
                    <p className="font-medium">{adapter.display_name || adapter.trigger_tag || adapter.name}</p>
                    <p className="truncate font-mono text-[10px] text-muted-foreground">{adapter.path}</p>
                  </div>
                ))}
              </div>
            )}
          </FieldGroup>
        </div>
      ),
    },
  ];

  return (
    <WizardShell
      title="Image LoRA Trainer"
      subtitle="Train MFLUX image-LoRAs voor album covers, artist styles en visual identities."
      steps={steps}
      step={step}
      onStepChange={setStep}
      onFinish={() => train.mutate()}
      isFinishing={train.isPending}
      finishLabel="Start training"
    />
  );
}
