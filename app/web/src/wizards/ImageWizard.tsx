import * as React from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { GitCompare, Image as ImageIcon, Loader2, Plus, UploadCloud, Video, Wand2, X } from "lucide-react";

import { WizardShell, FieldGroup, type WizardStepDef } from "@/components/wizard/WizardShell";
import { AIPromptStep } from "@/components/wizard/AIPromptStep";
import { ReviewStep } from "@/components/wizard/ReviewStep";
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
  getMfluxJob,
  getMfluxLoras,
  getMfluxModels,
  startMfluxJob,
  uploadMfluxImage,
  type MfluxJob,
  type MfluxLoraAdapter,
  type MfluxModel,
} from "@/lib/api";
import { useJobsStore } from "@/store/jobs";
import { useWizardStore } from "@/store/wizard";

type ImageAction = "generate" | "edit" | "inpaint" | "upscale" | "depth";

interface SelectedImageLora {
  path: string;
  name: string;
  scale: number;
  model_id?: string;
  family?: string;
}

interface UploadedImage {
  upload_id: string;
  filename: string;
  path: string;
  url: string;
}

const ACTIONS: Array<{ id: ImageAction; label: string; hint: string }> = [
  { id: "generate", label: "Text-to-image", hint: "Nieuw beeld uit prompt" },
  { id: "edit", label: "Edit / img2img", hint: "Bronbeeld + prompt" },
  { id: "inpaint", label: "Fill / inpaint", hint: "Bronbeeld + mask" },
  { id: "upscale", label: "Upscale", hint: "Vergroot bestaande art" },
  { id: "depth", label: "Depth map", hint: "Exporteer depth control" },
];

const MODE = "image" as const;

function imageAction(value: unknown): ImageAction {
  return ACTIONS.some((item) => item.id === value) ? (value as ImageAction) : "generate";
}

function numberValue(value: unknown, fallback: number) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

function adapterLabel(adapter: MfluxLoraAdapter) {
  return adapter.display_name || adapter.trigger_tag || adapter.name;
}

function supportsAction(model: MfluxModel | undefined, action: ImageAction) {
  const caps = new Set(model?.capabilities ?? []);
  if (action === "edit") return caps.has("edit") || caps.has("img2img") || caps.has("in_context_edit");
  return caps.has(action);
}

function compatibleLora(adapter: MfluxLoraAdapter, model?: MfluxModel) {
  if (!adapter.family || !model?.family) return true;
  return adapter.family === model.family;
}

function ImageUploadBox({
  label,
  description,
  image,
  onUploaded,
  onClear,
  required,
}: {
  label: string;
  description: string;
  image: UploadedImage | null;
  onUploaded: (image: UploadedImage) => void;
  onClear: () => void;
  required?: boolean;
}) {
  const inputRef = React.useRef<HTMLInputElement | null>(null);
  const upload = useMutation({
    mutationFn: uploadMfluxImage,
    onSuccess: (resp) => {
      if (!resp.success || !resp.path || !resp.url || !resp.upload_id || !resp.filename) {
        toast.error(resp.error || "Image upload mislukt");
        return;
      }
      onUploaded({
        upload_id: resp.upload_id,
        filename: resp.filename,
        path: resp.path,
        url: resp.url,
      });
      toast.success(`${label} geüpload.`);
    },
    onError: (error: Error) => toast.error(error.message),
  });

  return (
    <div className="rounded-md border bg-background/40 p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div>
          <Label>{label}{required ? " *" : ""}</Label>
          <p className="text-xs text-muted-foreground">{description}</p>
        </div>
        {image && (
          <Button type="button" variant="ghost" size="icon" onClick={onClear} aria-label={`Clear ${label}`}>
            <X className="size-4" />
          </Button>
        )}
      </div>
      {image ? (
        <div className="grid gap-3 sm:grid-cols-[160px_1fr]">
          <div className="overflow-hidden rounded-md border bg-black/70">
            <img src={image.url} alt={label} className="h-32 w-full object-contain" />
          </div>
          <div className="min-w-0 space-y-2">
            <p className="truncate text-sm font-medium">{image.filename}</p>
            <p className="break-all text-xs text-muted-foreground">{image.path}</p>
            <Button type="button" size="sm" variant="outline" onClick={() => inputRef.current?.click()}>
              Replace
            </Button>
          </div>
        </div>
      ) : (
        <button
          type="button"
          className="flex min-h-36 w-full flex-col items-center justify-center gap-2 rounded-md border border-dashed bg-muted/20 text-sm text-muted-foreground transition hover:bg-muted/40"
          onClick={() => inputRef.current?.click()}
        >
          {upload.isPending ? <Loader2 className="size-7 animate-spin" /> : <UploadCloud className="size-7" />}
          <span>{upload.isPending ? "Uploading..." : "Upload PNG, JPG or WEBP"}</span>
        </button>
      )}
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp,image/bmp,image/tiff"
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.currentTarget.value = "";
          if (file) upload.mutate(file);
        }}
      />
    </div>
  );
}

export function ImageWizard() {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const addJob = useJobsStore((s) => s.addJob);
  const updateJob = useJobsStore((s) => s.updateJob);
  const setDraft = useWizardStore((s) => s.setDraft);
  const draft = (useWizardStore((s) => s.drafts[MODE]) ?? {}) as Record<string, unknown>;
  const [step, setStep] = React.useState(0);
  const [aiPromptPending, setAiPromptPending] = React.useState(false);
  const [action, setAction] = React.useState<ImageAction>(() => imageAction(draft.action));
  const [prompt, setPrompt] = React.useState(() => String(draft.prompt || "Premium square music artwork, cinematic detail, no typography, no watermark"));
  const [modelId, setModelId] = React.useState(() => String(draft.model_id || "qwen-image"));
  const [width, setWidth] = React.useState(() => numberValue(draft.width, 1024));
  const [height, setHeight] = React.useState(() => numberValue(draft.height, 1024));
  const [steps, setSteps] = React.useState(() => numberValue(draft.steps, 30));
  const [seed, setSeed] = React.useState(() => String(draft.seed ?? "-1"));
  const [strength, setStrength] = React.useState(() => numberValue(draft.strength, 0.55));
  const [upscaleFactor, setUpscaleFactor] = React.useState(() => numberValue(draft.upscale_factor, 2));
  const [sourceImage, setSourceImage] = React.useState<UploadedImage | null>(() => (draft.source_image as UploadedImage | null) || null);
  const [maskImage, setMaskImage] = React.useState<UploadedImage | null>(() => (draft.mask_image as UploadedImage | null) || null);
  const [selectedAdapter, setSelectedAdapter] = React.useState("");
  const [loras, setLoras] = React.useState<SelectedImageLora[]>(() => Array.isArray(draft.lora_adapters) ? draft.lora_adapters as SelectedImageLora[] : []);
  const [jobId, setJobId] = React.useState("");
  const [job, setJob] = React.useState<MfluxJob | null>(null);

  const modelsQ = useQuery({ queryKey: ["mflux", "models"], queryFn: getMfluxModels, staleTime: 60_000 });
  const lorasQ = useQuery({ queryKey: ["mflux", "loras"], queryFn: getMfluxLoras, staleTime: 20_000 });
  const models = modelsQ.data?.models ?? [];
  const actionModels = modelsQ.data?.by_action?.[action] ?? models.filter((model) => supportsAction(model, action));
  const status = modelsQ.data?.status;
  const actionStatus = status?.action_readiness?.[action];
  const adapters = lorasQ.data?.adapters ?? [];
  const selectedModel = models.find((model) => model.id === modelId) ?? actionModels[0] ?? models[0];
  const compatibleAdapters = adapters.filter((adapter) => compatibleLora(adapter, selectedModel));
  const imageUrl = String(job?.result_summary?.image_url || job?.result?.image_url || job?.result?.url || "");
  const resultId = String(job?.result_summary?.result_id || job?.result?.result_id || job?.id || "");

  React.useEffect(() => {
    const defaultModel = modelsQ.data?.defaults?.[action] || actionModels[0]?.id;
    if (defaultModel && (!selectedModel || !supportsAction(selectedModel, action))) {
      setModelId(defaultModel);
    }
  }, [action, actionModels, modelsQ.data?.defaults, selectedModel]);

  React.useEffect(() => {
    if (!selectedModel) return;
    setSteps(selectedModel.default_steps || 20);
    setWidth(selectedModel.default_width || 1024);
    setHeight(selectedModel.default_height || 1024);
  }, [selectedModel?.id]);

  const needsSource = action !== "generate";
  const needsMask = action === "inpaint";
  const payload = {
    action,
    prompt: action === "upscale" || action === "depth" ? "" : prompt,
    model_id: modelId,
    width,
    height,
    steps,
    seed: seed.trim() ? Number(seed) : -1,
    quantize: selectedModel?.quantization_default ?? 8,
    image_upload_id: sourceImage?.upload_id || "",
    image_path: sourceImage?.path || "",
    image_url: sourceImage?.url || "",
    mask_upload_id: maskImage?.upload_id || "",
    mask_path: maskImage?.path || "",
    mask_url: maskImage?.url || "",
    strength,
    upscale_factor: upscaleFactor,
    lora_adapters: loras,
  };
  const canRender = status?.ready !== false && Boolean(modelId) && (!needsSource || Boolean(sourceImage?.path)) && (!needsMask || Boolean(maskImage?.path));
  const draftFingerprint = React.useRef("");

  React.useEffect(() => {
    const next = { ...payload, source_image: sourceImage, mask_image: maskImage };
    const json = JSON.stringify(next);
    if (json === draftFingerprint.current) return;
    draftFingerprint.current = json;
    const timer = window.setTimeout(() => setDraft(MODE, next), 250);
    return () => window.clearTimeout(timer);
  }, [payload, sourceImage, maskImage, setDraft]);

  const hydrateFromAi = (data: Record<string, unknown>) => {
    const nextAction = imageAction(data.action ?? action);
    setAction(nextAction);
    if (typeof data.prompt === "string") setPrompt(data.prompt);
    if (typeof data.model_id === "string") setModelId(data.model_id);
    setWidth(numberValue(data.width, width));
    setHeight(numberValue(data.height, height));
    setSteps(numberValue(data.steps, steps));
    setSeed(String(data.seed ?? seed));
    setStrength(numberValue(data.strength, strength));
    setUpscaleFactor(numberValue(data.upscale_factor, upscaleFactor));
    if (Array.isArray(data.lora_adapters)) setLoras(data.lora_adapters as SelectedImageLora[]);
    setDraft(MODE, { ...payload, ...data, action: nextAction });
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
        error: resp.job.error,
        metadata: resp.job as unknown as Record<string, unknown>,
      });
      if (!["succeeded", "failed", "error"].includes(String(resp.job.state))) {
        window.setTimeout(poll, 1800);
      } else if (resp.job.state === "succeeded") {
        qc.invalidateQueries({ queryKey: ["mflux"] });
      }
    };
    void poll();
    return () => {
      cancelled = true;
    };
  }, [jobId, qc, updateJob]);

  const start = useMutation({
    mutationFn: () => startMfluxJob(payload),
    onSuccess: (resp) => {
      if (!resp.success || !resp.job?.id) {
        toast.error(resp.error || "MFLUX job mislukt");
        return;
      }
      setJobId(resp.job.id);
      setJob(resp.job);
      addJob({
        id: resp.job.id,
        kind: "mflux",
        label: prompt.slice(0, 60) || "MFLUX image",
        progress: resp.job.progress || 0,
        status: resp.job.status || "queued",
        state: resp.job.state || "queued",
        stage: resp.job.stage,
        kindLabel: "MFLUX image",
        detailsPath: `/api/mflux/jobs/${encodeURIComponent(resp.job.id)}`,
        metadata: resp.job as unknown as Record<string, unknown>,
        startedAt: Date.now(),
      });
      toast.success("MFLUX image job gestart.");
      setStep(6);
    },
    onError: (error: Error) => toast.error(error.message),
  });

  const addLora = () => {
    const adapter = compatibleAdapters.find((item) => item.path === selectedAdapter);
    if (!adapter || loras.some((item) => item.path === adapter.path)) return;
    setLoras((items) => [
      ...items,
      { path: adapter.path, name: adapterLabel(adapter), scale: 0.75, model_id: adapter.model_id, family: adapter.family },
    ]);
    setSelectedAdapter("");
  };

  const stepsDef: WizardStepDef[] = [
    {
      key: "ai",
      title: "AI Fill",
      description: "Beschrijf het beeld of de edit; AI vult prompt, model, canvas, seed en LoRA-hints.",
      isValid: !aiPromptPending && (action === "upscale" || action === "depth" || prompt.trim().length > 6),
      render: () => (
        <AIPromptStep
          mode="image"
          placeholder="Bijv. 'album cover voor donkere west-coast rap, nachtelijke boulevard, cinematic, geen tekst'"
          examples={[
            "square album art, luxury noir studio portrait, no text",
            "edit this cover into a rainy neon street scene",
            "depth map setup for a dramatic close-up portrait",
          ]}
          onPendingChange={setAiPromptPending}
          onHydrated={hydrateFromAi}
        />
      ),
    },
    {
      key: "prompt",
      title: "Prompt",
      description: action === "upscale" || action === "depth" ? "Deze actie gebruikt vooral het bronbeeld." : "Beschrijf het beeld. Voor art: geen tekst/watermarks in het beeld.",
      isValid: action === "upscale" || action === "depth" || prompt.trim().length > 6,
      render: () => (
        <FieldGroup title="Image brief">
          <div className="space-y-2">
            <Label>Prompt</Label>
            <Textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={7}
              disabled={action === "upscale" || action === "depth"}
            />
          </div>
        </FieldGroup>
      ),
    },
    {
      key: "model",
      title: "Model & canvas",
      description: "MFLUX draait alleen op Apple Silicon + MLX.",
      isValid: canRender,
      render: () => (
        <div className="space-y-4">
          {status && !status.ready && (
            <p className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
              {status.blocking_reason || "MFLUX is niet klaar."}
            </p>
          )}
          {actionStatus && !actionStatus.ready && (
            <p className="rounded-md bg-amber-500/10 p-3 text-sm text-amber-200">
              Missing MFLUX command for {action}: {(actionStatus.missing_commands || []).join(", ") || actionStatus.reason}
            </p>
          )}
          <FieldGroup title="Render path">
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1.5">
                <Label>Actie</Label>
                <Select value={action} onValueChange={(value) => setAction(value as ImageAction)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {ACTIONS.map((item) => (
                      <SelectItem key={item.id} value={item.id}>{item.label} · {item.hint}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label>Model</Label>
                <Select value={modelId} onValueChange={setModelId}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {actionModels.map((model) => (
                      <SelectItem key={model.id} value={model.id}>{model.label} · {model.preset}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            {selectedModel && (
              <p className="text-xs text-muted-foreground">{selectedModel.description}</p>
            )}
          </FieldGroup>
          <FieldGroup title="Canvas">
            <div className="grid gap-3 sm:grid-cols-4">
              <div className="space-y-1.5">
                <Label>Width</Label>
                <Input type="number" value={width} onChange={(e) => setWidth(Number(e.target.value) || 1024)} disabled={action === "upscale" || action === "depth"} />
              </div>
              <div className="space-y-1.5">
                <Label>Height</Label>
                <Input type="number" value={height} onChange={(e) => setHeight(Number(e.target.value) || 1024)} disabled={action === "upscale" || action === "depth"} />
              </div>
              <div className="space-y-1.5">
                <Label>Steps</Label>
                <Input type="number" value={steps} onChange={(e) => setSteps(Number(e.target.value) || 20)} disabled={action === "upscale" || action === "depth"} />
              </div>
              <div className="space-y-1.5">
                <Label>Seed</Label>
                <Input value={seed} onChange={(e) => setSeed(e.target.value)} disabled={action === "upscale" || action === "depth"} />
              </div>
            </div>
            {(action === "edit" || action === "inpaint") && (
              <div className="mt-4 space-y-2">
                <Label>Strength</Label>
                <Slider min={0.05} max={1} step={0.05} value={[strength]} onValueChange={(value) => setStrength(value[0] ?? 0.55)} />
                <p className="text-xs text-muted-foreground">Current: {strength.toFixed(2)}</p>
              </div>
            )}
            {action === "upscale" && (
              <div className="mt-4 space-y-1.5">
                <Label>Upscale factor</Label>
                <Input type="number" min={2} max={4} value={upscaleFactor} onChange={(e) => setUpscaleFactor(Number(e.target.value) || 2)} />
              </div>
            )}
          </FieldGroup>
        </div>
      ),
    },
    {
      key: "canvas",
      title: "Input images",
      description: "Upload bronbeeld en mask waar nodig.",
      isValid: !needsSource || (Boolean(sourceImage?.path) && (!needsMask || Boolean(maskImage?.path))),
      render: () => (
        <div className="grid gap-4 lg:grid-cols-2">
          <ImageUploadBox
            label="Source image"
            description={needsSource ? "Nodig voor edit, inpaint, upscale en depth." : "Optioneel, alleen nodig voor niet-text acties."}
            image={sourceImage}
            onUploaded={setSourceImage}
            onClear={() => setSourceImage(null)}
            required={needsSource}
          />
          {action === "inpaint" ? (
            <ImageUploadBox
              label="Mask image"
              description="Wit gebied wordt gevuld/aangepast, zwart blijft beschermd."
              image={maskImage}
              onUploaded={setMaskImage}
              onClear={() => setMaskImage(null)}
              required
            />
          ) : (
            <div className="rounded-md border bg-background/30 p-4 text-sm text-muted-foreground">
              <GitCompare className="mb-2 size-5" />
              Na render zie je hier automatisch before/after wanneer een source image gebruikt is.
            </div>
          )}
        </div>
      ),
    },
    {
      key: "loras",
      title: "Image LoRAs",
      description: "Stapelen mag: elke LoRA krijgt een eigen scale.",
      isValid: true,
      render: () => (
        <FieldGroup title="LoRA stack">
          <div className="flex gap-2">
            <Select value={selectedAdapter} onValueChange={setSelectedAdapter}>
              <SelectTrigger><SelectValue placeholder="Kies image-LoRA" /></SelectTrigger>
              <SelectContent>
                {compatibleAdapters.map((adapter) => (
                  <SelectItem key={adapter.path} value={adapter.path}>{adapterLabel(adapter)}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button type="button" variant="outline" onClick={addLora} disabled={!selectedAdapter}>
              <Plus className="size-4" />
            </Button>
          </div>
          <div className="space-y-2">
            {loras.length === 0 && <p className="text-sm text-muted-foreground">Geen image-LoRA gekozen.</p>}
            {loras.map((lora, index) => (
              <div key={lora.path} className="rounded-md border bg-background/35 p-3">
                <div className="mb-2 flex items-center justify-between gap-2">
                  <span className="truncate text-sm font-medium">{lora.name}</span>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">{lora.scale.toFixed(2)}</Badge>
                    <Button
                      type="button"
                      size="icon"
                      variant="ghost"
                      onClick={() => setLoras((items) => items.filter((_, i) => i !== index))}
                      aria-label={`Remove ${lora.name}`}
                    >
                      <X className="size-4" />
                    </Button>
                  </div>
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
        </FieldGroup>
      ),
    },
    {
      key: "review",
      title: "Review",
      description: "Controleer de MFLUX payload voordat je rendert.",
      isValid: canRender,
      render: () => (
        <ReviewStep
          payload={payload}
          warnings={[
            ...(needsSource && !sourceImage ? ["Deze actie heeft eerst een source image nodig."] : []),
            ...(needsMask && !maskImage ? ["Inpaint heeft ook een mask image nodig."] : []),
          ]}
          primaryFields={[
            { key: "action", label: "Action" },
            { key: "model_id", label: "Model" },
            { key: "width", label: "Width" },
            { key: "height", label: "Height" },
            { key: "steps", label: "Steps" },
            { key: "seed", label: "Seed" },
            { key: "image_path", label: "Source" },
            { key: "mask_path", label: "Mask" },
          ]}
        />
      ),
    },
    {
      key: "result",
      title: "Result",
      description: "Preview, before/after en jobstatus.",
      isValid: true,
      render: () => (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <Button onClick={() => start.mutate()} disabled={start.isPending || !canRender}>
              {start.isPending ? <Loader2 className="size-4 animate-spin" /> : <Wand2 className="size-4" />}
              Render opnieuw
            </Button>
            {job && <Badge variant="outline">{job.stage || job.status || job.state}</Badge>}
          </div>
          <div className={sourceImage && imageUrl ? "grid gap-4 lg:grid-cols-2" : ""}>
            {sourceImage && imageUrl && (
              <div className="overflow-hidden rounded-xl border bg-card/40">
                <div className="border-b px-3 py-2 text-sm text-muted-foreground">Before</div>
                <img src={sourceImage.url} alt="Source" className="max-h-[680px] w-full object-contain" />
              </div>
            )}
            <div className="min-h-96 overflow-hidden rounded-xl border bg-card/40">
              {imageUrl ? (
                <>
                  <div className="border-b px-3 py-2 text-sm text-muted-foreground">After</div>
                  <img src={imageUrl} alt="MFLUX result" className="max-h-[680px] w-full object-contain" />
                </>
              ) : (
                <div className="flex min-h-96 flex-col items-center justify-center gap-2 text-muted-foreground">
                  <ImageIcon className="size-10" />
                  <span>Render verschijnt hier.</span>
                </div>
              )}
            </div>
          </div>
          {job?.error && <p className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">{job.error}</p>}
          {imageUrl && (
            <Button
              variant="outline"
              onClick={() => navigate("/wizard/video", {
                state: {
                  image_url: imageUrl,
                  title: prompt.slice(0, 48) || "Image animation",
                  prompt,
                  target_type: "image_result",
                  target_id: resultId,
                },
              })}
              className="gap-2"
            >
              <Video className="size-4" />
              Animate image
            </Button>
          )}
        </div>
      ),
    },
  ];

  return (
    <WizardShell
      title="Image Studio"
      subtitle="MFLUX generatie, edit, LoRA stack, upscale en depth in één MLX-first wizard."
      steps={stepsDef}
      step={step}
      onStepChange={setStep}
      onFinish={() => start.mutate()}
      isFinishing={start.isPending || aiPromptPending}
      finishLabel="Render image"
    />
  );
}
