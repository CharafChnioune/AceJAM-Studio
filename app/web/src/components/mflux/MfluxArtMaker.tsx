import * as React from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Image as ImageIcon, Loader2, Plus, RefreshCw, Wand2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { toast } from "@/components/ui/sonner";
import {
  attachMfluxArt,
  getMfluxJob,
  getMfluxLoras,
  getMfluxModels,
  startMfluxJob,
  type MfluxJob,
  type MfluxLoraAdapter,
  type MfluxModel,
} from "@/lib/api";
import { useJobsStore } from "@/store/jobs";

interface SelectedImageLora {
  path: string;
  name: string;
  scale: number;
  model_id?: string;
}

interface MfluxArtMakerProps {
  title?: string;
  artist?: string;
  context?: string;
  targetType?: "generation_result" | "song" | "album" | "album_family";
  targetId?: string;
  compact?: boolean;
}

function defaultPrompt(title?: string, artist?: string, context?: string) {
  const subject = [title, artist].filter(Boolean).join(" by ");
  return [
    subject ? `Square premium cover artwork for "${subject}"` : "Square premium music cover artwork",
    context,
    "cinematic composition, high-detail, no typography, no watermark, no logo",
  ].filter(Boolean).join(", ");
}

function adapterLabel(adapter: MfluxLoraAdapter) {
  return adapter.display_name || adapter.trigger_tag || adapter.name;
}

function supportsGenerate(model?: MfluxModel) {
  return Boolean(model?.capabilities?.includes("generate"));
}

function compatibleLora(adapter: MfluxLoraAdapter, model?: MfluxModel) {
  if (!adapter.family || !model?.family) return true;
  return adapter.family === model.family;
}

export function MfluxArtMaker({
  title,
  artist,
  context,
  targetType,
  targetId,
  compact = false,
}: MfluxArtMakerProps) {
  const qc = useQueryClient();
  const addJob = useJobsStore((s) => s.addJob);
  const updateJob = useJobsStore((s) => s.updateJob);
  const [prompt, setPrompt] = React.useState(defaultPrompt(title, artist, context));
  const [modelId, setModelId] = React.useState("qwen-image");
  const [width, setWidth] = React.useState(1024);
  const [height, setHeight] = React.useState(1024);
  const [steps, setSteps] = React.useState<number | undefined>(undefined);
  const [seed, setSeed] = React.useState("");
  const [selectedAdapter, setSelectedAdapter] = React.useState("");
  const [loras, setLoras] = React.useState<SelectedImageLora[]>([]);
  const [activeJobId, setActiveJobId] = React.useState("");
  const [activeJob, setActiveJob] = React.useState<MfluxJob | null>(null);

  React.useEffect(() => {
    setPrompt((current) => current || defaultPrompt(title, artist, context));
  }, [artist, context, title]);

  const modelsQ = useQuery({
    queryKey: ["mflux", "models"],
    queryFn: getMfluxModels,
    staleTime: 60_000,
  });
  const lorasQ = useQuery({
    queryKey: ["mflux", "loras"],
    queryFn: getMfluxLoras,
    staleTime: 20_000,
  });

  const models = modelsQ.data?.models ?? [];
  const generationModels = models.filter(supportsGenerate);
  const selectedModel = models.find((m) => m.id === modelId) ?? models[0];
  const adapters = (lorasQ.data?.adapters ?? []).filter((adapter) => compatibleLora(adapter, selectedModel));
  const status = modelsQ.data?.status;
  const imageUrl = String(activeJob?.result_summary?.image_url || activeJob?.result?.image_url || activeJob?.result?.url || "");
  const imagePath = String(activeJob?.result?.path || "");
  const canAttach = Boolean(targetType && targetId && activeJob?.result_summary?.result_id);

  React.useEffect(() => {
    if (!selectedModel) return;
    setSteps((current) => current ?? selectedModel.default_steps);
    setWidth((current) => current || selectedModel.default_width || 1024);
    setHeight((current) => current || selectedModel.default_height || 1024);
  }, [selectedModel]);

  React.useEffect(() => {
    if (!activeJobId) return;
    let stopped = false;
    const tick = async () => {
      try {
        const resp = await getMfluxJob(activeJobId);
        if (stopped || !resp.job) return;
        setActiveJob(resp.job);
        updateJob(activeJobId, {
          progress: resp.job.progress,
          status: resp.job.status || resp.job.state,
          state: resp.job.state,
          stage: resp.job.stage,
          metadata: resp.job as unknown as Record<string, unknown>,
          error: resp.job.error,
        });
        if (!["succeeded", "failed", "error"].includes(String(resp.job.state))) {
          window.setTimeout(tick, 1800);
        } else if (resp.job.state === "succeeded") {
          qc.invalidateQueries({ queryKey: ["mflux"] });
        }
      } catch (error) {
        toast.error((error as Error).message);
      }
    };
    void tick();
    return () => {
      stopped = true;
    };
  }, [activeJobId, qc, updateJob]);

  const startJob = useMutation({
    mutationFn: () =>
      startMfluxJob({
        action: "generate",
        title,
        prompt,
        model_id: modelId,
        width,
        height,
        steps,
        seed: seed.trim() ? Number(seed) : -1,
        quantize: selectedModel?.quantization_default ?? 8,
        lora_adapters: loras,
      }),
    onSuccess: (resp) => {
      if (!resp.success || !resp.job?.id) {
        toast.error(resp.error || "MFLUX job kon niet starten");
        return;
      }
      setActiveJobId(resp.job.id);
      setActiveJob(resp.job);
      addJob({
        id: resp.job.id,
        kind: "mflux",
        label: title || "MFLUX artwork",
        progress: resp.job.progress || 0,
        status: resp.job.status || "queued",
        state: resp.job.state || "queued",
        stage: resp.job.stage,
        kindLabel: "MFLUX image",
        detailsPath: `/api/mflux/jobs/${encodeURIComponent(resp.job.id)}`,
        metadata: resp.job as unknown as Record<string, unknown>,
        startedAt: Date.now(),
      });
      toast.success("MFLUX artwork gestart.");
    },
    onError: (error: Error) => toast.error(error.message),
  });

  const upscaleJob = useMutation({
    mutationFn: () =>
      startMfluxJob({
        action: "upscale",
        title: `${title || "art"} upscaled`,
        model_id: "seedvr2",
        image_path: imagePath,
        upscale_factor: 2,
      }),
    onSuccess: (resp) => {
      if (!resp.success || !resp.job?.id) {
        toast.error(resp.error || "MFLUX upscale kon niet starten");
        return;
      }
      setActiveJobId(resp.job.id);
      setActiveJob(resp.job);
      addJob({
        id: resp.job.id,
        kind: "mflux",
        label: `${title || "MFLUX artwork"} upscale`,
        progress: resp.job.progress || 0,
        status: resp.job.status || "queued",
        state: resp.job.state || "queued",
        stage: resp.job.stage,
        kindLabel: "MFLUX upscale",
        detailsPath: `/api/mflux/jobs/${encodeURIComponent(resp.job.id)}`,
        metadata: resp.job as unknown as Record<string, unknown>,
        startedAt: Date.now(),
      });
      toast.success("MFLUX upscale gestart.");
    },
    onError: (error: Error) => toast.error(error.message),
  });

  const attach = useMutation({
    mutationFn: () =>
      attachMfluxArt({
        source_result_id: activeJob?.result_summary?.result_id,
        target_type: targetType,
        target_id: targetId,
      }),
    onSuccess: (resp) => {
      if (!resp.success) {
        toast.error(resp.error || "Artwork koppelen mislukt");
        return;
      }
      toast.success("Artwork gekoppeld.");
      qc.invalidateQueries({ queryKey: ["community"] });
    },
    onError: (error: Error) => toast.error(error.message),
  });

  const addLora = () => {
    const adapter = adapters.find((item) => item.path === selectedAdapter);
    if (!adapter || loras.some((item) => item.path === adapter.path)) return;
    setLoras((items) => [
      ...items,
      { path: adapter.path, name: adapterLabel(adapter), scale: 0.75, model_id: adapter.model_id },
    ]);
    setSelectedAdapter("");
  };

  return (
    <section className="space-y-4 rounded-xl border bg-card/45 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <ImageIcon className="size-4 text-primary" />
            <h3 className="text-sm font-semibold">MFLUX Art Maker</h3>
            <Badge variant="outline" className="text-[10px]">MLX</Badge>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            Genereer song- of album-art met MFLUX, geen Ollama-image route.
          </p>
        </div>
        <Button size="sm" variant="ghost" onClick={() => void qc.invalidateQueries({ queryKey: ["mflux"] })}>
          <RefreshCw className="size-3.5" />
          Refresh
        </Button>
      </div>

      {status && !status.ready && (
        <p className="rounded-md bg-destructive/10 p-3 text-xs text-destructive">
          {status.blocking_reason || "MFLUX is niet klaar op deze machine."}
        </p>
      )}

      <div className={compact ? "space-y-3" : "grid gap-4 lg:grid-cols-[1.1fr_0.9fr]"}>
        <div className="space-y-3">
          <div className="space-y-1.5">
            <Label>Art prompt</Label>
            <Textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={4} />
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label>Model</Label>
              <Select value={modelId} onValueChange={setModelId}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {generationModels.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.label} · {model.preset}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label>Seed</Label>
              <Input value={seed} onChange={(e) => setSeed(e.target.value)} placeholder="-1 random" />
            </div>
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="space-y-1.5">
              <Label>Width</Label>
              <Input type="number" value={width} onChange={(e) => setWidth(Number(e.target.value) || 1024)} />
            </div>
            <div className="space-y-1.5">
              <Label>Height</Label>
              <Input type="number" value={height} onChange={(e) => setHeight(Number(e.target.value) || 1024)} />
            </div>
            <div className="space-y-1.5">
              <Label>Steps</Label>
              <Input type="number" value={steps ?? ""} onChange={(e) => setSteps(Number(e.target.value) || selectedModel?.default_steps)} />
            </div>
          </div>

          <div className="space-y-2">
            <Label>Image LoRAs</Label>
            <div className="flex gap-2">
              <Select value={selectedAdapter} onValueChange={setSelectedAdapter}>
                <SelectTrigger>
                  <SelectValue placeholder="Kies image-LoRA" />
                </SelectTrigger>
                <SelectContent>
                  {adapters.map((adapter) => (
                    <SelectItem key={adapter.path} value={adapter.path}>
                      {adapterLabel(adapter)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button type="button" variant="outline" onClick={addLora} disabled={!selectedAdapter}>
                <Plus className="size-3.5" />
              </Button>
            </div>
            {loras.map((lora, index) => (
              <div key={lora.path} className="rounded-md border bg-background/35 p-2">
                <div className="mb-2 flex items-center justify-between gap-2">
                  <span className="truncate text-xs font-medium">{lora.name}</span>
                  <Button
                    type="button"
                    size="sm"
                    variant="ghost"
                    onClick={() => setLoras((items) => items.filter((_, i) => i !== index))}
                  >
                    Remove
                  </Button>
                </div>
                <Slider
                  min={0}
                  max={1.5}
                  step={0.05}
                  value={[lora.scale]}
                  onValueChange={(value) => {
                    const scale = value[0] ?? 0.75;
                    setLoras((items) => items.map((item, i) => i === index ? { ...item, scale } : item));
                  }}
                />
              </div>
            ))}
          </div>

          <div className="flex flex-wrap gap-2">
            <Button onClick={() => startJob.mutate()} disabled={!prompt.trim() || startJob.isPending || status?.ready === false}>
              {startJob.isPending ? <Loader2 className="size-4 animate-spin" /> : <Wand2 className="size-4" />}
              Create art
            </Button>
            {canAttach && (
              <Button variant="outline" onClick={() => attach.mutate()} disabled={attach.isPending}>
                {attach.isPending ? <Loader2 className="size-4 animate-spin" /> : <ImageIcon className="size-4" />}
                Koppel artwork
              </Button>
            )}
            {imagePath && (
              <Button variant="outline" onClick={() => upscaleJob.mutate()} disabled={upscaleJob.isPending || status?.ready === false}>
                {upscaleJob.isPending ? <Loader2 className="size-4 animate-spin" /> : <RefreshCw className="size-4" />}
                Upscale
              </Button>
            )}
          </div>
        </div>

        <div className="min-h-64 overflow-hidden rounded-lg border bg-background/40">
          {imageUrl ? (
            <img src={imageUrl} alt="MFLUX artwork" className="h-full max-h-[520px] w-full object-contain" />
          ) : (
            <div className="flex min-h-64 flex-col items-center justify-center gap-2 p-6 text-center text-sm text-muted-foreground">
              <ImageIcon className="size-8" />
              <span>Artwork preview verschijnt hier.</span>
              {activeJob && (
                <Badge variant="outline">
                  {activeJob.stage || activeJob.status || activeJob.state}
                </Badge>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
