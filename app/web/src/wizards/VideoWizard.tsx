import * as React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Film, GitCompare, Loader2, Plus, UploadCloud, Video, Wand2, X } from "lucide-react";

import { WizardShell, FieldGroup, type WizardStepDef } from "@/components/wizard/WizardShell";
import { AIPromptStep } from "@/components/wizard/AIPromptStep";
import { ReviewStep } from "@/components/wizard/ReviewStep";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "@/components/ui/sonner";
import {
  attachMlxVideo,
  getMlxVideoJob,
  getMlxVideoLoras,
  getMlxVideoModels,
  startMlxVideoJob,
  uploadMlxVideoMedia,
  type MlxVideoJob,
  type MlxVideoLoraAdapter,
  type MlxVideoModel,
  type MlxVideoModelDir,
} from "@/lib/api";
import { useJobsStore } from "@/store/jobs";
import { useWizardStore } from "@/store/wizard";

type VideoAction = "t2v" | "i2v" | "a2v" | "song_video" | "final";

interface SelectedVideoLora {
  path: string;
  name: string;
  scale: number;
  role?: string;
  family?: string;
}

interface MediaSource {
  upload_id?: string;
  filename?: string;
  path: string;
  url: string;
  media_kind: "image" | "audio" | "video";
}

const ACTIONS: Array<{ id: VideoAction; label: string; hint: string }> = [
  { id: "t2v", label: "Text-to-video", hint: "Snel draften uit prompt" },
  { id: "i2v", label: "Image-to-video", hint: "Start vanuit art/foto" },
  { id: "a2v", label: "Audio-to-video", hint: "Clip op audio sturen" },
  { id: "song_video", label: "Song video", hint: "Muziekvideo uit track" },
  { id: "final", label: "Final/HQ", hint: "Zelfde seed/source groter renderen" },
];

const MODE = "video" as const;

function videoAction(value: unknown): VideoAction {
  return ACTIONS.some((item) => item.id === value) ? (value as VideoAction) : "t2v";
}

function numberValue(value: unknown, fallback: number) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

function adapterLabel(adapter: MlxVideoLoraAdapter) {
  return adapter.display_name || adapter.name || adapter.path.split("/").at(-1) || "Video LoRA";
}

function supportsAction(model: MlxVideoModel | undefined, action: VideoAction) {
  const caps = new Set(model?.capabilities ?? []);
  return caps.has(action) || (action === "song_video" && caps.has("a2v"));
}

function compatibleLora(adapter: MlxVideoLoraAdapter, model?: MlxVideoModel) {
  if (!adapter.family || !model?.family) return true;
  return adapter.family === model.family;
}

function isImageSource(source: MediaSource | null) {
  return source?.media_kind === "image";
}

function isAudioSource(source: MediaSource | null) {
  return source?.media_kind === "audio";
}

function SourceUploadBox({
  label,
  description,
  accept,
  source,
  onUploaded,
  onClear,
  required,
}: {
  label: string;
  description: string;
  accept: string;
  source: MediaSource | null;
  onUploaded: (source: MediaSource) => void;
  onClear: () => void;
  required?: boolean;
}) {
  const inputRef = React.useRef<HTMLInputElement | null>(null);
  const upload = useMutation({
    mutationFn: uploadMlxVideoMedia,
    onSuccess: (resp) => {
      if (!resp.success || !resp.path || !resp.url || !resp.upload_id) {
        toast.error(resp.error || "Upload mislukt");
        return;
      }
      const kind = resp.media_kind === "audio" || resp.media_kind === "video" ? resp.media_kind : "image";
      onUploaded({
        upload_id: resp.upload_id,
        filename: resp.filename || "media",
        path: resp.path,
        url: resp.url,
        media_kind: kind,
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
        {source && (
          <Button type="button" variant="ghost" size="icon" onClick={onClear} aria-label={`Clear ${label}`}>
            <X className="size-4" />
          </Button>
        )}
      </div>
      {source ? (
        <div className="grid gap-3 sm:grid-cols-[180px_1fr]">
          <div className="overflow-hidden rounded-md border bg-black/70">
            {source.media_kind === "image" ? (
              <img src={source.url} alt={label} className="h-32 w-full object-contain" />
            ) : source.media_kind === "audio" ? (
              <div className="flex h-32 items-center justify-center p-3">
                <audio src={source.url} controls className="w-full" />
              </div>
            ) : (
              <video src={source.url} controls className="h-32 w-full object-contain" />
            )}
          </div>
          <div className="min-w-0 space-y-2">
            <p className="truncate text-sm font-medium">{source.filename || source.url}</p>
            <p className="break-all text-xs text-muted-foreground">{source.path || source.url}</p>
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
          <span>{upload.isPending ? "Uploading..." : "Upload source media"}</span>
        </button>
      )}
      <input
        ref={inputRef}
        type="file"
        accept={accept}
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

function sourceFromRouteState(state: unknown): {
  action?: VideoAction;
  image?: MediaSource;
  audio?: MediaSource;
  prompt?: string;
  title?: string;
  targetType?: string;
  targetId?: string;
  sourceJobId?: string;
} {
  if (!state || typeof state !== "object") return {};
  const raw = state as Record<string, unknown>;
  const audioUrl = String(raw.audio_url || raw.audioUrl || "");
  const imageUrl = String(raw.image_url || raw.imageUrl || raw.art_url || "");
  const prompt = String(raw.prompt || raw.caption || raw.title || "");
  return {
    action: audioUrl ? "song_video" : imageUrl ? "i2v" : undefined,
    audio: audioUrl ? { url: audioUrl, path: audioUrl, media_kind: "audio", filename: String(raw.title || "song") } : undefined,
    image: imageUrl ? { url: imageUrl, path: imageUrl, media_kind: "image", filename: String(raw.title || "image") } : undefined,
    prompt,
    title: String(raw.title || ""),
    targetType: String(raw.target_type || ""),
    targetId: String(raw.target_id || raw.result_id || raw.song_id || raw.album_id || ""),
    sourceJobId: String(raw.source_job_id || raw.draft_job_id || ""),
  };
}

export function VideoWizard() {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const route = useLocation();
  const initial = React.useMemo(() => sourceFromRouteState(route.state), [route.state]);
  const addJob = useJobsStore((s) => s.addJob);
  const updateJob = useJobsStore((s) => s.updateJob);
  const setDraft = useWizardStore((s) => s.setDraft);
  const draft = (useWizardStore((s) => s.drafts[MODE]) ?? {}) as Record<string, unknown>;
  const restoredPrompt = String(draft.prompt || "");
  const routePrompt = String(initial.prompt || "");
  const [step, setStep] = React.useState(0);
  const [aiPromptPending, setAiPromptPending] = React.useState(false);
  const [action, setAction] = React.useState<VideoAction>(initial.action || videoAction(draft.action));
  const [prompt, setPrompt] = React.useState(
    restoredPrompt || (routePrompt
      ? `${routePrompt}, cinematic real-life music video, natural motion, professional camera, no text, no watermark`
      : "Cinematic real-life music video, handheld camera, natural motion, moody lighting, no text, no watermark")
  );
  const [modelId, setModelId] = React.useState(() => String(draft.model_id || "ltx2-fast-draft"));
  const [modelDir, setModelDir] = React.useState(() => String(draft.model_dir || ""));
  const [width, setWidth] = React.useState(() => numberValue(draft.width, 512));
  const [height, setHeight] = React.useState(() => numberValue(draft.height, 320));
  const [frames, setFrames] = React.useState(() => numberValue(draft.num_frames ?? draft.frames, 33));
  const [fps, setFps] = React.useState(() => numberValue(draft.fps, 24));
  const [steps, setSteps] = React.useState(() => numberValue(draft.steps, 8));
  const [seed, setSeed] = React.useState(() => String(draft.seed ?? "-1"));
  const [guideScale, setGuideScale] = React.useState(() => String(draft.guide_scale || ""));
  const [shift, setShift] = React.useState(() => String(draft.shift || ""));
  const [sourceImage, setSourceImage] = React.useState<MediaSource | null>(initial.image || (draft.source_image as MediaSource | null) || null);
  const [endImage, setEndImage] = React.useState<MediaSource | null>((draft.end_image as MediaSource | null) || null);
  const [sourceAudio, setSourceAudio] = React.useState<MediaSource | null>(initial.audio || (draft.source_audio as MediaSource | null) || null);
  const [enhancePrompt, setEnhancePrompt] = React.useState(() => Boolean(draft.enhance_prompt));
  const [spatialUpscaler, setSpatialUpscaler] = React.useState(() => String(draft.spatial_upscaler || ""));
  const [tiling, setTiling] = React.useState(() => Boolean(draft.tiling));
  const [targetType] = React.useState(initial.targetType || "");
  const [targetId] = React.useState(initial.targetId || "");
  const [finalSourceJobId] = React.useState(initial.sourceJobId || "");
  const [selectedAdapter, setSelectedAdapter] = React.useState("");
  const [loras, setLoras] = React.useState<SelectedVideoLora[]>(() => Array.isArray(draft.lora_adapters) ? draft.lora_adapters as SelectedVideoLora[] : []);
  const [jobId, setJobId] = React.useState("");
  const [job, setJob] = React.useState<MlxVideoJob | null>(null);

  const modelsQ = useQuery({ queryKey: ["mlx-video", "models"], queryFn: getMlxVideoModels, staleTime: 60_000 });
  const lorasQ = useQuery({ queryKey: ["mlx-video", "loras"], queryFn: getMlxVideoLoras, staleTime: 20_000 });
  const status = modelsQ.data?.status;
  const models = modelsQ.data?.models ?? [];
  const actionModels = modelsQ.data?.by_action?.[action] ?? models.filter((model) => supportsAction(model, action));
  const selectedModel = models.find((model) => model.id === modelId) ?? actionModels[0] ?? models[0];
  const modelDirs = modelsQ.data?.registered_model_dirs ?? status?.registered_model_dirs ?? [];
  const adapters = lorasQ.data?.adapters ?? [];
  const compatibleAdapters = adapters.filter((adapter) => compatibleLora(adapter, selectedModel));
  const videoUrl = String(job?.result_summary?.primary_video_url || job?.result_summary?.video_url || job?.result?.primary_video_url || job?.result?.video_url || job?.result?.url || "");
  const rawVideoUrl = String(job?.result_summary?.raw_video_url || job?.result?.raw_video_url || "");
  const posterUrl = String(job?.result_summary?.poster_url || job?.result?.poster_url || "");
  const resultId = String(job?.result_summary?.result_id || job?.result?.result_id || "");
  const sourceJobId = String(job?.id || "");
  const ltxCaps = status?.command_help?.ltx?.capabilities ?? {};
  const ltxEndFrameSupported = Boolean(ltxCaps.end_image || status?.patch_status?.pr23_ltx_i2v_end_frame);

  React.useEffect(() => {
    const defaultModel = modelsQ.data?.defaults?.[action] || actionModels[0]?.id;
    if (defaultModel && (!selectedModel || !supportsAction(selectedModel, action))) {
      setModelId(defaultModel);
    }
  }, [action, actionModels, modelsQ.data?.defaults, selectedModel]);

  React.useEffect(() => {
    if (!selectedModel) return;
    setWidth(selectedModel.default_width || 512);
    setHeight(selectedModel.default_height || 320);
    setFrames(selectedModel.default_frames || 33);
    setFps(selectedModel.default_fps || 24);
    setSteps(selectedModel.default_steps || 8);
    setGuideScale(selectedModel.guide_scale === undefined ? "" : String(selectedModel.guide_scale));
    setShift(selectedModel.shift === undefined ? "" : String(selectedModel.shift));
    if (selectedModel.requires_model_dir) {
      const firstCompatible = modelDirs.find((dir) => !selectedModel.family || dir.family === selectedModel.family) ?? modelDirs[0];
      setModelDir((current) => current || firstCompatible?.path || "");
    } else {
      setModelDir("");
    }
  }, [selectedModel?.id, modelDirs]);

  const needsImage = action === "i2v";
  const needsAudio = action === "a2v" || action === "song_video";
  const needsModelDir = Boolean(selectedModel?.requires_model_dir);
  const endImageBlocked = Boolean(endImage) && selectedModel?.engine === "ltx" && !ltxEndFrameSupported;
  const payload = {
    action,
    prompt,
    title: initial.title || "MLX video",
    model_id: modelId,
    model_dir: needsModelDir ? modelDir : "",
    width,
    height,
    num_frames: frames,
    fps,
    steps,
    seed: seed.trim() ? Number(seed) : -1,
    guide_scale: guideScale,
    shift,
    image_path: sourceImage?.path || "",
    source_image_url: sourceImage?.url || "",
    end_image_path: endImage?.path || "",
    source_end_image_url: endImage?.url || "",
    audio_path: sourceAudio?.path || "",
    audio_url: sourceAudio?.url || "",
    enhance_prompt: enhancePrompt,
    spatial_upscaler: spatialUpscaler,
    tiling,
    lora_adapters: loras,
    source_job_id: action === "final" ? (finalSourceJobId || sourceJobId) : "",
    target_type: targetType,
    target_id: targetId,
    audio_policy: action === "song_video" ? "replace_with_source" : "none",
    mux_audio: action === "song_video",
  };
  const canRender =
    status?.ready !== false &&
    Boolean(modelId) &&
    prompt.trim().length > 6 &&
    (!needsImage || Boolean(sourceImage?.path || sourceImage?.url)) &&
    (!needsAudio || Boolean(sourceAudio?.path || sourceAudio?.url)) &&
    (!needsModelDir || Boolean(modelDir)) &&
    !endImageBlocked;
  const draftFingerprint = React.useRef("");

  React.useEffect(() => {
    const next = { ...payload, source_image: sourceImage, end_image: endImage, source_audio: sourceAudio };
    const json = JSON.stringify(next);
    if (json === draftFingerprint.current) return;
    draftFingerprint.current = json;
    const timer = window.setTimeout(() => setDraft(MODE, next), 250);
    return () => window.clearTimeout(timer);
  }, [payload, sourceImage, endImage, sourceAudio, setDraft]);

  const hydrateFromAi = (data: Record<string, unknown>) => {
    const nextAction = videoAction(data.action ?? action);
    setAction(nextAction);
    if (typeof data.prompt === "string") setPrompt(data.prompt);
    if (typeof data.model_id === "string") setModelId(data.model_id);
    if (typeof data.model_dir === "string") setModelDir(data.model_dir);
    setWidth(numberValue(data.width, width));
    setHeight(numberValue(data.height, height));
    setFrames(numberValue(data.num_frames ?? data.frames, frames));
    setFps(numberValue(data.fps, fps));
    setSteps(numberValue(data.steps, steps));
    setSeed(String(data.seed ?? seed));
    setGuideScale(String(data.guide_scale ?? guideScale ?? ""));
    setShift(String(data.shift ?? shift ?? ""));
    setEnhancePrompt(Boolean(data.enhance_prompt));
    setSpatialUpscaler(String(data.spatial_upscaler || ""));
    setTiling(Boolean(data.tiling));
    if (Array.isArray(data.lora_adapters)) setLoras(data.lora_adapters as SelectedVideoLora[]);
    setDraft(MODE, { ...payload, ...data, action: nextAction });
  };

  React.useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    const poll = async () => {
      const resp = await getMlxVideoJob(jobId);
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
        window.setTimeout(poll, 2500);
      } else if (resp.job.state === "succeeded") {
        qc.invalidateQueries({ queryKey: ["mlx-video"] });
      }
    };
    void poll();
    return () => {
      cancelled = true;
    };
  }, [jobId, qc, updateJob]);

  const start = useMutation({
    mutationFn: (override?: Record<string, unknown>) => startMlxVideoJob({ ...payload, ...(override || {}) }),
    onSuccess: (resp) => {
      if (!resp.success || !resp.job?.id) {
        toast.error(resp.error || "MLX video job mislukt");
        return;
      }
      setJobId(resp.job.id);
      setJob(resp.job);
      addJob({
        id: resp.job.id,
        kind: "mlx-video",
        label: prompt.slice(0, 60) || "MLX video",
        progress: resp.job.progress || 0,
        status: resp.job.status || "queued",
        state: resp.job.state || "queued",
        stage: resp.job.stage,
        kindLabel: "MLX video",
        detailsPath: `/api/mlx-video/jobs/${encodeURIComponent(resp.job.id)}`,
        metadata: resp.job as unknown as Record<string, unknown>,
        startedAt: Date.now(),
      });
      toast.success("MLX video job gestart.");
      setStep(6);
    },
    onError: (error: Error) => toast.error(error.message),
  });

  const addLora = () => {
    const adapter = compatibleAdapters.find((item) => item.path === selectedAdapter);
    if (!adapter || loras.some((item) => item.path === adapter.path)) return;
    setLoras((items) => [
      ...items,
      { path: adapter.path, name: adapterLabel(adapter), scale: 1, role: adapter.role || "shared", family: adapter.family },
    ]);
    setSelectedAdapter("");
  };

  const attach = useMutation({
    mutationFn: (target: { target_type: string; target_id: string }) =>
      attachMlxVideo({
        source_result_id: resultId,
        result_id: resultId,
        target_type: target.target_type,
        target_id: target.target_id,
      }),
    onSuccess: (resp) => {
      if (!resp.success) {
        toast.error(resp.error || "Video koppelen mislukt");
        return;
      }
      toast.success("Video gekoppeld.");
      qc.invalidateQueries({ queryKey: ["mlx-video", "attachments"] });
    },
    onError: (error: Error) => toast.error(error.message),
  });

  const modelDirOptions = (dirs: MlxVideoModelDir[]) =>
    dirs.filter((dir) => !selectedModel?.family || !dir.family || dir.family === selectedModel.family);

  const stepsDef: WizardStepDef[] = [
    {
      key: "ai",
      title: "AI Fill",
      description: "Beschrijf de clip; AI vult action, cinematic prompt, draft settings en source-intent.",
      isValid: !aiPromptPending && prompt.trim().length > 6,
      render: () => (
        <AIPromptStep
          mode="video"
          placeholder="Bijv. 'short real-life music video for dark rap song, neon street, handheld camera, small draft'"
          examples={[
            "song_video for a moody rap track, night city, close-up performance shots",
            "image-to-video: make the album cover slowly breathe with cinematic camera motion",
            "fast draft for sunny beach performance clip, natural handheld movement",
          ]}
          onPendingChange={setAiPromptPending}
          onHydrated={hydrateFromAi}
        />
      ),
    },
    {
      key: "prompt",
      title: "Video brief",
      description: "Beschrijf beweging, camera en sfeer. Draft-first: eerst klein, daarna final.",
      isValid: prompt.trim().length > 6,
      render: () => (
        <FieldGroup title="Prompt">
          <div className="space-y-2">
            <Label>Prompt</Label>
            <Textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={7} />
          </div>
        </FieldGroup>
      ),
    },
    {
      key: "source",
      title: "Source media",
      description: "Voor Image-to-video of Song video voeg je bronbeeld/audio toe.",
      isValid: (!needsImage || isImageSource(sourceImage)) && (!needsAudio || isAudioSource(sourceAudio)),
      render: () => (
        <div className="space-y-4">
          <FieldGroup title="Action">
            <Select value={action} onValueChange={(value) => setAction(value as VideoAction)}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                {ACTIONS.map((item) => (
                  <SelectItem key={item.id} value={item.id}>{item.label} · {item.hint}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </FieldGroup>
          <div className="grid gap-4 lg:grid-cols-3">
            <SourceUploadBox
              label="Source image"
              description={needsImage ? "Nodig voor image-to-video." : "Optioneel als startframe voor LTX of Wan I2V."}
              accept="image/png,image/jpeg,image/webp,image/bmp,image/tiff"
              source={sourceImage}
              onUploaded={setSourceImage}
              onClear={() => setSourceImage(null)}
              required={needsImage}
            />
            <SourceUploadBox
              label="End frame"
              description={
                ltxEndFrameSupported
                  ? "Optioneel: PR #23 first+last-frame conditioning voor LTX I2V."
                  : "Optioneel, maar geblokkeerd totdat Install/Update PR #23 support heeft."
              }
              accept="image/png,image/jpeg,image/webp,image/bmp,image/tiff"
              source={endImage}
              onUploaded={setEndImage}
              onClear={() => setEndImage(null)}
            />
            <SourceUploadBox
              label="Source audio / song"
              description={needsAudio ? "Nodig voor audio/song video." : "Optioneel voor LTX audio-to-video."}
              accept="audio/wav,audio/mpeg,audio/flac,audio/mp4,audio/aac,audio/ogg"
              source={sourceAudio}
              onUploaded={setSourceAudio}
              onClear={() => setSourceAudio(null)}
              required={needsAudio}
            />
          </div>
          {endImageBlocked && (
            <p className="rounded-md bg-amber-500/10 p-3 text-sm text-amber-200">
              End frame gebruikt <code>--end-image</code>. Run eerst Install/Update zodat PR #23 clean gepatcht wordt, of haal het end frame weg.
            </p>
          )}
        </div>
      ),
    },
    {
      key: "model",
      title: "Model & preset",
      description: "LTX is direct draftbaar; Wan vereist een geregistreerde geconverteerde model-dir.",
      isValid: canRender,
      render: () => (
        <div className="space-y-4">
          {status && !status.ready && (
            <p className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
              {status.blocking_reason || "MLX Video Studio is niet klaar. Run Install/Update voor app/video-env."}
            </p>
          )}
          <FieldGroup title="Preset">
            <div className="grid gap-3 sm:grid-cols-2">
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
              <div className="space-y-1.5">
                <Label>Wan model-dir</Label>
                <Select value={modelDir || "__none"} onValueChange={(value) => setModelDir(value === "__none" ? "" : value)} disabled={!needsModelDir}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__none">Geen model-dir</SelectItem>
                    {modelDirOptions(modelDirs).map((dir) => (
                      <SelectItem key={dir.path} value={dir.path}>{dir.label} · {dir.family || "wan"}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            {selectedModel && <p className="text-xs text-muted-foreground">{selectedModel.description}</p>}
            {needsModelDir && !modelDir && (
              <p className="rounded-md bg-amber-500/10 p-3 text-xs text-amber-200">
                Registreer eerst een geconverteerde Wan MLX model-dir in Settings → Video.
              </p>
            )}
          </FieldGroup>
          <FieldGroup title="Draft settings">
            <div className="grid gap-3 sm:grid-cols-3">
              <div className="space-y-1.5">
                <Label>Width</Label>
                <Input type="number" value={width} onChange={(event) => setWidth(Number(event.target.value) || 512)} />
              </div>
              <div className="space-y-1.5">
                <Label>Height</Label>
                <Input type="number" value={height} onChange={(event) => setHeight(Number(event.target.value) || 320)} />
              </div>
              <div className="space-y-1.5">
                <Label>Frames</Label>
                <Input type="number" value={frames} onChange={(event) => setFrames(Number(event.target.value) || 33)} />
              </div>
              <div className="space-y-1.5">
                <Label>FPS</Label>
                <Input type="number" value={fps} onChange={(event) => setFps(Number(event.target.value) || 24)} />
              </div>
              <div className="space-y-1.5">
                <Label>Steps</Label>
                <Input type="number" value={steps} onChange={(event) => setSteps(Number(event.target.value) || 8)} />
              </div>
              <div className="space-y-1.5">
                <Label>Seed</Label>
                <Input value={seed} onChange={(event) => setSeed(event.target.value)} />
              </div>
              <div className="space-y-1.5">
                <Label>Guide scale</Label>
                <Input value={guideScale} onChange={(event) => setGuideScale(event.target.value)} placeholder="auto" />
              </div>
              <div className="space-y-1.5">
                <Label>Shift</Label>
                <Input value={shift} onChange={(event) => setShift(event.target.value)} placeholder="auto" />
              </div>
            </div>
          </FieldGroup>
          <FieldGroup title="Command features">
            <div className="grid gap-3 sm:grid-cols-3">
              <label className="flex items-center justify-between gap-3 rounded-md border bg-background/35 p-3 text-sm">
                <span>Prompt enhancer</span>
                <Switch checked={enhancePrompt} onCheckedChange={setEnhancePrompt} disabled={selectedModel?.engine !== "ltx"} />
              </label>
              <label className="flex items-center justify-between gap-3 rounded-md border bg-background/35 p-3 text-sm">
                <span>Tiling</span>
                <Switch checked={tiling} onCheckedChange={setTiling} />
              </label>
              <div className="space-y-1.5">
                <Label>Spatial upscaler</Label>
                <Select value={spatialUpscaler || "__none"} onValueChange={(value) => setSpatialUpscaler(value === "__none" ? "" : value)} disabled={selectedModel?.engine !== "ltx"}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__none">Off</SelectItem>
                    <SelectItem value="latent">Latent 2x</SelectItem>
                    <SelectItem value="pixel">Pixel 2x</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge variant={ltxEndFrameSupported ? "default" : "outline"}>
                LTX end frame {ltxEndFrameSupported ? "ready" : "needs PR #23"}
              </Badge>
              <Badge variant="outline">Output flag {status?.command_help?.[selectedModel?.engine || "ltx"]?.output_flag || "--output-path"}</Badge>
            </div>
          </FieldGroup>
        </div>
      ),
    },
    {
      key: "loras",
      title: "Video LoRAs",
      description: "Wan high/low LoRA's en LTX LoRA's kunnen mee in de render.",
      isValid: true,
      render: () => (
        <FieldGroup title="LoRA stack">
          <div className="flex gap-2">
            <Select value={selectedAdapter} onValueChange={setSelectedAdapter}>
              <SelectTrigger><SelectValue placeholder="Kies video-LoRA" /></SelectTrigger>
              <SelectContent>
                {compatibleAdapters.map((adapter) => (
                  <SelectItem key={adapter.path} value={adapter.path}>{adapterLabel(adapter)} · {adapter.role || "shared"}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button type="button" variant="outline" onClick={addLora} disabled={!selectedAdapter}>
              <Plus className="size-4" />
            </Button>
          </div>
          <div className="space-y-2">
            {loras.length === 0 && <p className="text-sm text-muted-foreground">Geen video-LoRA gekozen.</p>}
            {loras.map((lora, index) => (
              <div key={lora.path} className="rounded-md border bg-background/35 p-3">
                <div className="mb-2 flex items-center justify-between gap-2">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium">{lora.name}</p>
                    <p className="truncate font-mono text-[10px] text-muted-foreground">{lora.path}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">{lora.role || "shared"} · {lora.scale.toFixed(2)}</Badge>
                    <Button type="button" size="icon" variant="ghost" onClick={() => setLoras((items) => items.filter((_, i) => i !== index))}>
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
                    const scale = value[0] ?? 1;
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
      description: "Controleer de command payload voordat je een lange render start.",
      isValid: canRender,
      render: () => (
        <ReviewStep
          payload={payload}
          warnings={[
            ...(needsImage && !sourceImage ? ["Image-to-video heeft een source image nodig."] : []),
            ...(needsAudio && !sourceAudio ? ["Audio/song video heeft source audio nodig."] : []),
            ...(needsModelDir && !modelDir ? ["Wan preset heeft een geregistreerde model-dir nodig."] : []),
            ...(endImageBlocked ? ["End-frame conditioning vereist PR #23 / --end-image support."] : []),
            ...(frames > 81 ? ["Veel frames kan lang duren. Draft klein houden, daarna final maken."] : []),
          ]}
          primaryFields={[
            { key: "action", label: "Action" },
            { key: "model_id", label: "Model" },
            { key: "width", label: "Width" },
            { key: "height", label: "Height" },
            { key: "num_frames", label: "Frames" },
            { key: "steps", label: "Steps" },
            { key: "seed", label: "Seed" },
            { key: "image_path", label: "Image" },
            { key: "end_image_path", label: "End frame" },
            { key: "audio_path", label: "Audio" },
            { key: "enhance_prompt", label: "Enhance" },
            { key: "spatial_upscaler", label: "Upscaler" },
            { key: "tiling", label: "Tiling" },
          ]}
        />
      ),
    },
    {
      key: "result",
      title: "Result",
      description: "Speel de MP4 direct af. Als de draft klopt kun je final maken.",
      isValid: true,
      render: () => (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <Button onClick={() => start.mutate({})} disabled={start.isPending || !canRender}>
              {start.isPending ? <Loader2 className="size-4 animate-spin" /> : <Wand2 className="size-4" />}
              Render draft
            </Button>
            {videoUrl && (
              <Button
                variant="outline"
                onClick={() => start.mutate({ action: "final", model_id: "ltx2-final-hq", source_job_id: sourceJobId, target_type: targetType, target_id: targetId })}
                disabled={start.isPending}
              >
                <Film className="size-4" />
                Make Final
              </Button>
            )}
            {videoUrl && targetType && targetId && (
              <Button
                variant="outline"
                onClick={() => attach.mutate({ target_type: targetType, target_id: targetId })}
                disabled={attach.isPending || !resultId}
              >
                Koppel aan {targetType}
              </Button>
            )}
            {videoUrl && (
              <Button
                variant="ghost"
                onClick={() => attach.mutate({ target_type: "library", target_id: resultId || sourceJobId })}
                disabled={attach.isPending || !resultId}
              >
                Bewaar in library
              </Button>
            )}
            {job && <Badge variant="outline">{job.stage || job.status || job.state}</Badge>}
          </div>
          {sourceImage && videoUrl && (
            <div className="grid gap-4 lg:grid-cols-2">
              <div className="overflow-hidden rounded-xl border bg-card/40">
                <div className="border-b px-3 py-2 text-sm text-muted-foreground">Source</div>
                <img src={sourceImage.url} alt="Source" className="max-h-[480px] w-full object-contain" />
              </div>
              <div className="overflow-hidden rounded-xl border bg-card/40">
                <div className="border-b px-3 py-2 text-sm text-muted-foreground">Video</div>
                <video src={videoUrl} poster={posterUrl || undefined} controls className="max-h-[480px] w-full bg-black object-contain" />
              </div>
            </div>
          )}
          {!sourceImage && (
            <div className="min-h-96 overflow-hidden rounded-xl border bg-card/40">
              {videoUrl ? (
                <video src={videoUrl} poster={posterUrl || undefined} controls className="max-h-[680px] w-full bg-black object-contain" />
              ) : (
                <div className="flex min-h-96 flex-col items-center justify-center gap-2 text-muted-foreground">
                  <Video className="size-10" />
                  <span>Video verschijnt hier.</span>
                </div>
              )}
            </div>
          )}
          {sourceAudio && (
            <FieldGroup title="Audio source">
              <audio src={sourceAudio.url} controls className="w-full" />
              {action === "song_video" && (
                <p className="text-xs text-muted-foreground">
                  Song video policy: de gegenereerde video-audio wordt gestript en deze source audio wordt in de MP4 gemuxed.
                </p>
              )}
            </FieldGroup>
          )}
          {action === "song_video" && rawVideoUrl && rawVideoUrl !== videoUrl && (
            <p className="text-xs text-muted-foreground">
              Primary MP4 gebruikt de muxed source-audio. Raw generated clip blijft alleen bewaard voor audit/debug.
            </p>
          )}
          {job?.error && <p className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">{job.error}</p>}
          {videoUrl && (
            <div className="flex flex-wrap gap-2">
              <Button variant="outline" onClick={() => navigate("/library")}>Open library</Button>
              <Button variant="ghost" asChild>
                <a href={videoUrl} download>Download MP4</a>
              </Button>
            </div>
          )}
          {videoUrl && sourceImage && (
            <p className="flex items-center gap-2 text-xs text-muted-foreground">
              <GitCompare className="size-3.5" />
              Gebruik Make Final pas als beweging, compositie en seed goed voelen.
            </p>
          )}
        </div>
      ),
    },
  ];

  return (
    <WizardShell
      title="Video Studio"
      subtitle="MLX-video draft-first generatie, image/audio/song-to-video, Wan/LTX presets en video-LoRAs."
      steps={stepsDef}
      step={step}
      onStepChange={setStep}
      onFinish={() => start.mutate({})}
      isFinishing={start.isPending || aiPromptPending}
      finishLabel="Render video"
    />
  );
}
