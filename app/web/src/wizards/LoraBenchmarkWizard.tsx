import * as React from "react";
import { useNavigate } from "react-router-dom";
import { useForm, Controller } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { BarChart3, ListMusic, Music4, RefreshCw } from "lucide-react";

import { WizardShell, FieldGroup, type WizardStepDef } from "@/components/wizard/WizardShell";
import { AIPromptStep } from "@/components/wizard/AIPromptStep";
import { ReviewStep } from "@/components/wizard/ReviewStep";
import { GenerationAudioList } from "@/components/wizard/GenerationAudioList";
import { RenderInsightPanel } from "@/components/wizard/RenderInsightPanel";
import { AutomationFields } from "@/components/wizard/AutomationFields";
import { TagInput } from "@/components/wizard/TagInput";
import { AudioStyleSelector } from "@/components/wizard/AudioStyleSelector";
import { AudioBackendSelector } from "@/components/wizard/AudioBackendSelector";
import { AceStepAdvancedSettings } from "@/components/wizard/AceStepAdvancedSettings";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { toast } from "@/components/ui/sonner";
import {
  getLoraAdapters,
  getLoraSweepJob,
  startLoraSweepJob,
  type LoraAdapter,
  type LoraSweepJob,
} from "@/lib/api";
import { customSchema, simpleDefaults, type CustomFormValues } from "@/lib/schemas";
import {
  ACE_STEP_ADVANCED_DEFAULTS,
  ACE_STEP_ADVANCED_PAYLOAD_FIELDS,
  ACE_STEP_KEY_SCALE_OPTIONS,
  ACE_STEP_TIME_SIGNATURE_OPTIONS,
  OFFICIAL_AUDIO_FORMAT_OPTIONS,
} from "@/lib/aceStepSettings";
import { ACE_STEP_LANGUAGE_OPTIONS } from "@/lib/languages";
import { isGenerationLoraAdapter, loraAdapterLabel, loraTriggerOptions } from "@/lib/lora";
import { audioBackendLabel, useMlxDitForAudioBackend } from "@/lib/audioBackend";
import { mergeWizardDraft, useWizardDraft } from "@/hooks/useWizardDraft";
import { useWizardStore } from "@/store/wizard";
import { useJobsStore } from "@/store/jobs";
import { formatDuration } from "@/lib/utils";

const MODE = "lora_sweep" as const;

const SONG_MODELS = [
  ["acestep-v15-xl-sft", "ACE-Step v1.5 XL SFT (aanbevolen)"],
  ["acestep-v15-xl-base", "ACE-Step v1.5 XL Base"],
  ["acestep-v15-xl-turbo", "ACE-Step v1.5 XL Turbo"],
  ["acestep-v15-sft", "ACE-Step v1.5 SFT"],
  ["acestep-v15-base", "ACE-Step v1.5 Base"],
  ["acestep-v15-turbo", "ACE-Step v1.5 Turbo"],
  ["acestep-v15-turbo-shift1", "ACE-Step v1.5 Turbo (shift 1)"],
] as const;

const QUALITY_PROFILES = [
  ["draft", "Laag (docs-correct, volledig)"],
  ["standard", "Middel (docs standaard)"],
  ["chart_master", "Hoog (beste standaardkwaliteit)"],
] as const;

const TAG_SUGGESTIONS = [
  "rap", "hip hop", "trap", "drill", "boom bap", "g-funk", "pop", "rnb",
  "house", "techno", "cinematic", "rock", "hard drums", "808", "sub bass",
  "piano", "guitar", "strings", "analog synth", "clear English vocal",
  "dry upfront vocal", "wide stereo", "polished mix",
];

const NEG_SUGGESTIONS = [
  "mumbled vocals", "muddy mix", "weak drums", "washed out mix", "thin bass",
  "off-key vocals", "clipping", "low quality",
];

const TERMINAL_STATES = new Set(["succeeded", "success", "complete", "completed", "failed", "error", "stopped"]);

function docsCorrectRenderDefaults(songModel: string) {
  if (songModel.includes("turbo")) return { inference_steps: 8, shift: 3 };
  return { inference_steps: 64, shift: 3 };
}

function isExportedGenerationLora(adapter: LoraAdapter) {
  const source = String(adapter.source || "").toLowerCase();
  const path = String(adapter.path || "");
  return isGenerationLoraAdapter(adapter) && (source === "exports" || path.includes("/data/loras/"));
}

function sweepJobPatch(job: LoraSweepJob) {
  return {
    id: job.id,
    kind: "lora-sweep" as const,
    label: job.sweep_title || "LoRA Sweep",
    progress: job.progress || 0,
    status: job.status || job.state || "queued",
    state: job.state || "queued",
    stage: job.stage || "",
    kindLabel: "LoRA sweep",
    detailsPath: `/api/lora/sweeps/jobs/${encodeURIComponent(job.id)}`,
    logPath: `/api/lora/sweeps/jobs/${encodeURIComponent(job.id)}/log`,
    metadata: job as unknown as Record<string, unknown>,
    error: job.error || job.errors?.[0] || "",
    startedAt: job.created_at ? new Date(job.created_at).getTime() : Date.now(),
    updatedAt: job.updated_at || job.finished_at || "",
  };
}

export function LoraSweepWizard() {
  const navigate = useNavigate();
  const addJob = useJobsStore((s) => s.addJob);
  const openJob = useJobsStore((s) => s.openJob);
  const setResult = useWizardStore((s) => s.setResult);
  const lastResult = useWizardStore((s) => s.lastResult[MODE]);
  const warnings = useWizardStore((s) => s.warnings[MODE]) ?? [];
  const draft = useWizardStore((s) => s.drafts[MODE]);
  const defaults = React.useMemo<CustomFormValues>(
    () => ({
      ...simpleDefaults,
      task_type: "text2music",
      title: "LoRA Sweep",
      inference_steps: 64,
      guidance_scale: 8,
      shift: 3,
      audio_format: "wav32",
      batch_size: 1,
      use_lora: false,
      lora_adapter_path: "",
      lora_adapter_name: "",
      ...(ACE_STEP_ADVANCED_DEFAULTS as Partial<CustomFormValues>),
    }),
    [],
  );

  const form = useForm<CustomFormValues>({
    resolver: zodResolver(customSchema),
    defaultValues: mergeWizardDraft<CustomFormValues>(defaults, draft),
    mode: "onChange",
  });

  const [step, setStep] = React.useState(0);
  const [isStarting, setIsStarting] = React.useState(false);
  const [includeBaseline, setIncludeBaseline] = React.useState(false);
  const [loraScale, setLoraScale] = React.useState(1);
  const [activeJob, setActiveJob] = React.useState<LoraSweepJob | null>(null);
  const values = form.watch();
  const draftState = useWizardDraft(MODE, form);

  const adaptersQuery = useQuery({
    queryKey: ["lora", "sweep-adapters"],
    queryFn: getLoraAdapters,
    refetchInterval: 10_000,
  });
  const adapters = React.useMemo(
    () => (adaptersQuery.data?.adapters || []).filter(isExportedGenerationLora),
    [adaptersQuery.data?.adapters],
  );
  const totalRenders = adapters.length * values.batch_size + (includeBaseline ? values.batch_size : 0);

  React.useEffect(() => {
    if (!activeJob?.id || TERMINAL_STATES.has(String(activeJob.state || "").toLowerCase())) return;
    const timer = window.setInterval(async () => {
      try {
        const resp = await getLoraSweepJob(activeJob.id);
        if (resp.job) {
          setActiveJob(resp.job);
          addJob(sweepJobPatch(resp.job));
          if (TERMINAL_STATES.has(String(resp.job.state || "").toLowerCase())) {
            setResult(MODE, resp.job as unknown as Record<string, unknown>);
          }
        }
      } catch {
        // Job tracker will surface detail fetch errors; keep the wizard calm.
      }
    }, 2500);
    return () => window.clearInterval(timer);
  }, [activeJob?.id, activeJob?.state, addJob, setResult]);

  const setAdvancedValue = (key: string, value: unknown) => {
    form.setValue(key as keyof CustomFormValues, value as never, {
      shouldDirty: true,
      shouldValidate: true,
    });
  };

  const hydrate = (payload: Record<string, unknown>) => {
    const next: Partial<CustomFormValues> = {};
    for (const [k, v] of Object.entries(payload)) {
      if (k in form.getValues() || k === "simple_description") {
        // @ts-expect-error dynamic AI payload hydration
        next[k] = v;
      }
    }
    next.use_lora = false;
    next.lora_adapter_path = "";
    next.lora_adapter_name = "";
    const merged = { ...form.getValues(), ...next };
    form.reset(merged);
    draftState.saveNow(merged);
    return merged;
  };

  const buildPayload = () => {
    const v = form.getValues();
    const advanced: Record<string, unknown> = {};
    for (const key of ACE_STEP_ADVANCED_PAYLOAD_FIELDS) {
      const value = v[key as keyof CustomFormValues];
      if (value === undefined || value === "") continue;
      advanced[key] = value;
    }
    return {
      task_type: v.task_type,
      title: v.title,
      artist_name: v.artist_name,
      caption: v.caption,
      style_profile: v.style_profile,
      tags: v.tags,
      negative_tags: v.negative_tags,
      lyrics: v.instrumental ? "[Instrumental]" : v.lyrics,
      instrumental: v.instrumental,
      audio_duration: v.duration,
      duration: v.duration,
      bpm: v.bpm,
      key_scale: v.key_scale,
      time_signature: v.time_signature,
      vocal_language: v.vocal_language,
      song_model: v.song_model,
      audio_backend: v.audio_backend,
      use_mlx_dit: useMlxDitForAudioBackend(v.audio_backend),
      quality_profile: v.quality_profile,
      seed: v.seed,
      inference_steps: v.inference_steps,
      guidance_scale: v.guidance_scale,
      shift: v.shift,
      audio_format: v.audio_format,
      batch_size: v.batch_size,
      variant_count: v.batch_size,
      use_lora: false,
      lora_adapter_path: "",
      lora_adapter_name: "",
      use_lora_trigger: false,
      lora_trigger_tag: "",
      lora_scale: 0,
      auto_song_art: v.auto_song_art,
      auto_album_art: false,
      auto_video_clip: v.auto_video_clip,
      art_prompt: v.art_prompt,
      video_prompt: v.video_prompt,
      ...advanced,
    };
  };

  const handleFinish = async () => {
    setIsStarting(true);
    try {
      const payload = buildPayload();
      const resp = await startLoraSweepJob({
        sweep_title: values.title || "LoRA Sweep",
        render_payload: payload,
        variant_count: values.batch_size,
        include_baseline: includeBaseline,
        lora_scale: loraScale,
        trigger_mode: "auto",
        stop_on_error: false,
      });
      if (!resp.success || !resp.job_id) throw new Error(resp.error || "LoRA Sweep starten mislukt");
      const job = resp.job || ({ id: resp.job_id, sweep_title: values.title || "LoRA Sweep", state: "queued" } as LoraSweepJob);
      setActiveJob(job);
      setResult(MODE, job as unknown as Record<string, unknown>);
      addJob(sweepJobPatch(job));
      setStep(6);
      toast.success("LoRA Sweep gestart.");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "LoRA Sweep starten mislukt");
    } finally {
      setIsStarting(false);
    }
  };

  const steps: WizardStepDef[] = [
    {
      key: "ai",
      title: "AI song fill",
      description: "Laat AI één complete song-payload maken; de sweep rendert daarna elke LoRA met exact die song.",
      isValid: true,
      render: () => (
        <AIPromptStep
          mode="custom"
          placeholder="Maak een volledige rap/pop/club song met caption, tags, lyrics en metadata voor een LoRA sweep..."
          currentPayload={buildPayload()}
          onHydrated={hydrate}
        />
      ),
    },
    {
      key: "song",
      title: "Song",
      isValid: true,
      render: () => (
        <div className="space-y-4">
          <FieldGroup title="Naam">
            <div className="grid gap-3 sm:grid-cols-3">
              <div className="space-y-1.5">
                <Label>Titel</Label>
                <Input {...form.register("title")} />
              </div>
              <div className="space-y-1.5">
                <Label>Artiest</Label>
                <Input {...form.register("artist_name")} />
              </div>
              <div className="space-y-1.5">
                <Label>Taal</Label>
                <Controller
                  control={form.control}
                  name="vocal_language"
                  render={({ field }) => (
                    <Select value={field.value} onValueChange={field.onChange}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {ACE_STEP_LANGUAGE_OPTIONS.map(([code, name]) => (
                          <SelectItem key={code} value={code}>{name}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>
            </div>
          </FieldGroup>
          <FieldGroup title="Vocals">
            <div className="flex items-center justify-between rounded-md border bg-background/35 p-3">
              <div>
                <Label>Instrumentaal</Label>
                <p className="text-xs text-muted-foreground">Aan = lyrics worden vervangen door [Instrumental].</p>
              </div>
              <Controller
                control={form.control}
                name="instrumental"
                render={({ field }) => <Switch checked={field.value} onCheckedChange={field.onChange} />}
              />
            </div>
          </FieldGroup>
        </div>
      ),
    },
    {
      key: "sound",
      title: "Sound & lyrics",
      isValid: true,
      render: () => (
        <div className="space-y-4">
          <FieldGroup title="Caption">
            <AudioStyleSelector
              value={values.style_profile}
              onChange={(value) => form.setValue("style_profile", value, { shouldValidate: true })}
            />
            <Textarea rows={2} {...form.register("caption")} />
          </FieldGroup>
          <FieldGroup title="Tags">
            <Controller
              control={form.control}
              name="tags"
              render={({ field }) => (
                <TagInput value={field.value ?? ""} onChange={field.onChange} suggestions={TAG_SUGGESTIONS} />
              )}
            />
          </FieldGroup>
          <FieldGroup title="Negative tags">
            <Controller
              control={form.control}
              name="negative_tags"
              render={({ field }) => (
                <TagInput value={field.value ?? ""} onChange={field.onChange} suggestions={NEG_SUGGESTIONS} variant="negative" />
              )}
            />
          </FieldGroup>
          {!values.instrumental && (
            <FieldGroup title="Lyrics">
              <Textarea rows={16} className="font-mono text-xs leading-relaxed" {...form.register("lyrics")} />
            </FieldGroup>
          )}
        </div>
      ),
    },
    {
      key: "render",
      title: "Render & sweep",
      isValid: true,
      render: () => (
        <div className="space-y-4">
          <FieldGroup title="LoRA Sweep" description="De selector is weg: AceJAM rendert automatisch elke audio-LoRA in app/data/loras.">
            <div className="grid gap-3 sm:grid-cols-4">
              <div className="rounded-md border bg-background/35 p-3">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Audio-LoRAs</p>
                <p className="mt-1 font-mono text-lg">{adapters.length}</p>
              </div>
              <div className="rounded-md border bg-background/35 p-3">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Variaties</p>
                <p className="mt-1 font-mono text-lg">{values.batch_size}</p>
              </div>
              <div className="rounded-md border bg-background/35 p-3">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Baseline</p>
                <p className="mt-1 font-mono text-lg">{includeBaseline ? "aan" : "uit"}</p>
              </div>
              <div className="rounded-md border bg-background/35 p-3">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Totaal</p>
                <p className="mt-1 font-mono text-lg">{totalRenders}</p>
              </div>
            </div>
            <div className="mt-3 flex items-center justify-between rounded-md border bg-background/35 p-3">
              <div>
                <Label>No-LoRA baseline</Label>
                <p className="text-xs text-muted-foreground">Optioneel, standaard uit.</p>
              </div>
              <Switch checked={includeBaseline} onCheckedChange={setIncludeBaseline} />
            </div>
          </FieldGroup>

          <FieldGroup title="Model & kwaliteit">
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1.5">
                <Label>Fallback song model</Label>
                <Controller
                  control={form.control}
                  name="song_model"
                  render={({ field }) => (
                    <Select
                      value={field.value}
                      onValueChange={(nextModel) => {
                        field.onChange(nextModel);
                        const next = docsCorrectRenderDefaults(nextModel);
                        form.setValue("inference_steps", next.inference_steps);
                        form.setValue("shift", next.shift);
                      }}
                    >
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {SONG_MODELS.map(([id, label]) => <SelectItem key={id} value={id}>{label}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>
              <div className="space-y-1.5">
                <Label>Kwaliteit</Label>
                <Controller
                  control={form.control}
                  name="quality_profile"
                  render={({ field }) => (
                    <Select value={field.value} onValueChange={field.onChange}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {QUALITY_PROFILES.map(([id, label]) => <SelectItem key={id} value={id}>{label}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>
              <AudioBackendSelector
                value={values.audio_backend}
                onChange={(value) => form.setValue("audio_backend", value, { shouldValidate: true })}
              />
            </div>
          </FieldGroup>

          <AutomationFields control={form.control} register={form.register} values={values} />

          <FieldGroup title="Variaties en inference">
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-3">
                <div className="flex items-baseline justify-between">
                  <Label>Variaties per LoRA</Label>
                  <span className="font-mono text-xs">{values.batch_size}</span>
                </div>
                <Controller
                  control={form.control}
                  name="batch_size"
                  render={({ field }) => (
                    <Slider value={[field.value]} min={1} max={8} step={1} onValueChange={(v) => field.onChange(v[0] ?? 1)} />
                  )}
                />
              </div>
              <div className="space-y-3">
                <div className="flex items-baseline justify-between">
                  <Label>LoRA scale</Label>
                  <span className="font-mono text-xs">{loraScale.toFixed(2)}</span>
                </div>
                <Slider value={[loraScale]} min={0} max={1} step={0.05} onValueChange={(v) => setLoraScale(v[0] ?? 1)} />
              </div>
              <div className="space-y-3">
                <div className="flex items-baseline justify-between">
                  <Label>Inference steps</Label>
                  <span className="font-mono text-xs">{values.inference_steps}</span>
                </div>
                <Controller control={form.control} name="inference_steps" render={({ field }) => <Slider value={[field.value]} min={4} max={100} step={1} onValueChange={(v) => field.onChange(v[0])} />} />
              </div>
              <div className="space-y-3">
                <div className="flex items-baseline justify-between">
                  <Label>Guidance scale</Label>
                  <span className="font-mono text-xs">{values.guidance_scale.toFixed(1)}</span>
                </div>
                <Controller control={form.control} name="guidance_scale" render={({ field }) => <Slider value={[field.value]} min={1} max={15} step={0.1} onValueChange={(v) => field.onChange(v[0])} />} />
              </div>
              <div className="space-y-1.5">
                <Label>Audio format</Label>
                <Controller
                  control={form.control}
                  name="audio_format"
                  render={({ field }) => (
                    <Select value={field.value} onValueChange={field.onChange}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {OFFICIAL_AUDIO_FORMAT_OPTIONS.map(([value, label]) => <SelectItem key={value} value={value}>{label}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>
              <div className="space-y-1.5">
                <Label>Seed</Label>
                <Input type="number" placeholder="-1 voor deterministic sweep seeds" {...form.register("seed", { valueAsNumber: true })} />
              </div>
            </div>
          </FieldGroup>

          <FieldGroup title="BPM, key & maatsoort">
            <div className="grid gap-3 sm:grid-cols-3">
              <Input type="number" min={40} max={300} placeholder="BPM auto" {...form.register("bpm", { valueAsNumber: true })} />
              <Controller
                control={form.control}
                name="key_scale"
                render={({ field }) => (
                  <Select value={field.value || "auto"} onValueChange={(value) => field.onChange(value === "auto" ? undefined : value)}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {ACE_STEP_KEY_SCALE_OPTIONS.map((value) => <SelectItem key={value} value={value}>{value === "auto" ? "Auto" : value}</SelectItem>)}
                    </SelectContent>
                  </Select>
                )}
              />
              <Controller
                control={form.control}
                name="time_signature"
                render={({ field }) => (
                  <Select value={field.value || "auto"} onValueChange={(value) => field.onChange(value === "auto" ? undefined : value)}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {ACE_STEP_TIME_SIGNATURE_OPTIONS.map(([value, label]) => <SelectItem key={value || "auto"} value={value || "auto"}>{label}</SelectItem>)}
                    </SelectContent>
                  </Select>
                )}
              />
            </div>
          </FieldGroup>

          <FieldGroup title="Duur">
            <div className="flex items-baseline justify-between">
              <Label>Duur</Label>
              <span className="font-mono text-sm">{formatDuration(values.duration)}</span>
            </div>
            <Controller control={form.control} name="duration" render={({ field }) => <Slider value={[field.value ?? 60]} min={20} max={600} step={5} onValueChange={(v) => field.onChange(v[0] ?? 60)} />} />
          </FieldGroup>

          <FieldGroup title="Official ACE-Step controls">
            <AceStepAdvancedSettings values={values} onChange={setAdvancedValue} />
          </FieldGroup>
        </div>
      ),
    },
    {
      key: "adapters",
      title: "Adapters",
      isValid: adapters.length > 0 || includeBaseline,
      render: () => (
        <FieldGroup title="Gevonden audio-LoRAs" description="Alleen LoRA adapters die fysiek in app/data/loras staan en generation-loadable zijn.">
          <div className="flex items-center justify-between gap-3">
            <p className="text-sm text-muted-foreground">
              {adapters.length} LoRA{adapters.length === 1 ? "" : "s"} klaar voor sweep.
            </p>
            <Button type="button" variant="outline" size="sm" onClick={() => void adaptersQuery.refetch()} className="gap-2">
              <RefreshCw className="size-4" />
              Ververs
            </Button>
          </div>
          <div className="mt-3 max-h-96 space-y-2 overflow-auto pr-1">
            {adapters.map((adapter) => {
              const triggers = loraTriggerOptions(adapter);
              return (
                <div key={adapter.path} className="rounded-md border bg-background/35 p-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium">{loraAdapterLabel(adapter)}</p>
                      <p className="truncate text-xs text-muted-foreground">{adapter.path}</p>
                    </div>
                    <div className="flex flex-wrap items-center gap-1">
                      <Badge variant="outline">{adapter.song_model || adapter.model_variant || "auto model"}</Badge>
                      <Badge variant="secondary">{triggers[0] || "no trigger"}</Badge>
                    </div>
                  </div>
                </div>
              );
            })}
            {!adapters.length && (
              <p className="rounded-md border border-amber-500/30 bg-amber-500/10 p-3 text-sm text-amber-900 dark:text-amber-100">
                Geen generation-loadable audio-LoRAs gevonden in app/data/loras.
              </p>
            )}
          </div>
        </FieldGroup>
      ),
    },
    {
      key: "review",
      title: "Review & start",
      isValid: form.formState.isValid && (adapters.length > 0 || includeBaseline),
      render: () => (
        <div className="space-y-4">
          <ReviewStep
            payload={{
              sweep_title: values.title,
              variant_count: values.batch_size,
              include_baseline: includeBaseline,
              lora_scale: loraScale,
              total_renders: totalRenders,
              render_payload: buildPayload(),
            }}
            warnings={warnings}
            primaryFields={[
              { key: "sweep_title", label: "Sweep" },
              { key: "variant_count", label: "Variaties per LoRA" },
              { key: "include_baseline", label: "Baseline" },
              { key: "lora_scale", label: "LoRA scale" },
              { key: "total_renders", label: "Totaal renders" },
              { key: "render_payload.song_model", label: "Fallback model" },
              { key: "render_payload.audio_backend", label: "Backend", format: audioBackendLabel },
            ]}
          />
          <RenderInsightPanel payload={buildPayload()} warnings={warnings} />
        </div>
      ),
    },
    {
      key: "result",
      title: "Sweep resultaat",
      isValid: true,
      hidden: !activeJob && !lastResult,
      render: () => {
        const job = activeJob || (lastResult as unknown as LoraSweepJob | undefined);
        const rows = job?.items?.length ? job.items : job?.results || [];
        return (
          <div className="space-y-4">
            {job && (
              <FieldGroup title={job.sweep_title || "LoRA Sweep"}>
                <div className="grid gap-3 sm:grid-cols-4">
                  <div className="rounded-md border bg-background/35 p-3">
                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Status</p>
                    <p className="mt-1 text-sm">{job.status || job.state}</p>
                  </div>
                  <div className="rounded-md border bg-background/35 p-3">
                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Progress</p>
                    <p className="mt-1 font-mono text-sm">{job.progress || 0}%</p>
                  </div>
                  <div className="rounded-md border bg-background/35 p-3">
                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Audios</p>
                    <p className="mt-1 font-mono text-sm">{job.generated_audio_count || 0} / {job.expected_audio_count || 0}</p>
                  </div>
                  <div className="rounded-md border bg-background/35 p-3">
                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Groups</p>
                    <p className="mt-1 font-mono text-sm">{job.completed_items || 0} / {job.total_items || 0}</p>
                  </div>
                </div>
                {job.id && (
                  <Button type="button" variant="outline" className="mt-3 gap-2" onClick={() => openJob(job.id)}>
                    <ListMusic className="size-4" />
                    Open jobdetails
                  </Button>
                )}
              </FieldGroup>
            )}
            <div className="space-y-3">
              {rows.map((row, index) => {
                const variants = Array.isArray(row.variants) ? row.variants : [];
                const readyVariants = variants.filter((variant) => String(variant.state || "").toLowerCase() === "succeeded").length;
                const totalVariants = row.variant_count || variants.length || values.batch_size;
                return (
                  <div key={`${row.item_id || index}-${row.adapter_path || row.adapter_name}`} className="rounded-md border bg-background/35 p-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="min-w-0">
                        <p className="truncate text-sm font-medium">{row.adapter_name || `LoRA ${index + 1}`}</p>
                        <p className="truncate text-xs text-muted-foreground">
                          {row.song_model || "auto model"} · {readyVariants}/{totalVariants} varianten klaar
                        </p>
                      </div>
                      <Badge variant={String(row.state).toLowerCase() === "failed" ? "destructive" : String(row.state).toLowerCase() === "succeeded" ? "default" : "outline"}>
                        {row.status || row.state || "queued"}
                      </Badge>
                    </div>
                    {variants.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {variants.map((variant) => {
                          const state = String(variant.state || "").toLowerCase();
                          return (
                            <Badge
                              key={`${row.item_id || index}-variant-${variant.variant_index || variant.seed}`}
                              variant={state === "failed" ? "destructive" : state === "succeeded" ? "default" : state === "running" ? "secondary" : "outline"}
                              className="font-mono text-[10px]"
                            >
                              v{variant.variant_index || "?"}: {variant.variant_seed || variant.seed || "seed"} · {variant.status || variant.state || "queued"}
                            </Badge>
                          );
                        })}
                      </div>
                    )}
                    {row.error && <p className="mt-2 rounded-md bg-destructive/10 p-2 text-xs text-destructive">{row.error}</p>}
                    {row.result && (
                      <GenerationAudioList
                        result={row.result as Record<string, unknown>}
                        title={row.adapter_name || `LoRA ${index + 1}`}
                        artist="LoRA Sweep"
                        className="mt-3 space-y-2"
                      />
                    )}
                  </div>
                );
              })}
            </div>
            <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="flex flex-wrap items-center gap-2">
              <Button variant="outline" onClick={() => navigate("/library")} className="gap-2">
                <Music4 className="size-4" />
                Open library
              </Button>
              <Button variant="ghost" onClick={() => { form.reset(defaults); draftState.clear(); setActiveJob(null); setStep(0); }}>
                Nieuwe sweep
              </Button>
            </motion.div>
          </div>
        );
      },
    },
  ];

  return (
    <WizardShell
      title="LoRA Sweep"
      subtitle="Custom-style song bouwen, daarna automatisch renderen met elke audio-LoRA in de LoRA-map."
      steps={steps}
      step={step}
      onStepChange={setStep}
      onFinish={handleFinish}
      isFinishing={isStarting || Boolean(activeJob && !TERMINAL_STATES.has(String(activeJob.state || "").toLowerCase()))}
      finishLabel={isStarting ? "Start…" : "Start LoRA Sweep"}
    />
  );
}
