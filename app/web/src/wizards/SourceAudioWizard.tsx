import * as React from "react";
import { useNavigate } from "react-router-dom";
import { useForm, Controller } from "react-hook-form";
import { motion } from "framer-motion";
import { Music4, Video } from "lucide-react";

import { WizardShell, FieldGroup, type WizardStepDef } from "@/components/wizard/WizardShell";
import { AIPromptStep } from "@/components/wizard/AIPromptStep";
import { ReviewStep } from "@/components/wizard/ReviewStep";
import { GenerationJobStatus } from "@/components/wizard/GenerationJobStatus";
import { GenerationAudioList, firstGenerationAudioUrl } from "@/components/wizard/GenerationAudioList";
import { LoraSelector } from "@/components/wizard/LoraSelector";
import { RenderInsightPanel } from "@/components/wizard/RenderInsightPanel";
import { SourceAudioStep, type SourceAudioValue } from "@/components/wizard/SourceAudioStep";
import { TagInput } from "@/components/wizard/TagInput";
import { AudioStyleSelector } from "@/components/wizard/AudioStyleSelector";
import { AudioBackendSelector } from "@/components/wizard/AudioBackendSelector";
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
import { type WizardMode, api } from "@/lib/api";
import { DEFAULT_LORA_SCALE, normalizeLoraSelection, type LoraSelection } from "@/lib/lora";
import { audioBackendLabel, useMlxDitForAudioBackend } from "@/lib/audioBackend";
import { useGenerationJobRunner } from "@/hooks/useGenerationJobRunner";
import { mergeWizardDraft, useWizardDraft } from "@/hooks/useWizardDraft";
import { useWizardStore } from "@/store/wizard";
import { toast } from "@/components/ui/sonner";
import { cn, formatDuration } from "@/lib/utils";

const TASK_TYPE_BY_MODE: Record<string, string> = {
  cover: "cover",
  repaint: "repaint",
  extract: "extract",
  lego: "lego",
  complete: "complete",
};

const STEM_OPTIONS = [
  "vocals",
  "drums",
  "bass",
  "guitar",
  "keyboard",
  "synth",
  "strings",
  "brass",
  "fx",
];

const SONG_MODELS = [
  ["acestep-v15-xl-sft", "ACE-Step v1.5 XL SFT (aanbevolen)"],
  ["acestep-v15-xl-base", "ACE-Step v1.5 XL Base"],
  ["acestep-v15-xl-turbo", "ACE-Step v1.5 XL Turbo"],
  ["acestep-v15-sft", "ACE-Step v1.5 SFT"],
  ["acestep-v15-base", "ACE-Step v1.5 Base"],
  ["acestep-v15-turbo", "ACE-Step v1.5 Turbo"],
  ["acestep-v15-turbo-shift1", "ACE-Step v1.5 Turbo (shift 1)"],
] as const;

const BASE_ONLY_VARIANTS = new Set(["extract", "lego", "complete"]);

interface BaseSourceForm {
  task_type: string;
  title: string;
  artist_name: string;
  caption: string;
  style_profile: string;
  tags: string;
  negative_tags: string;
  lyrics: string;
  instrumental: boolean;
  duration: number;
  bpm?: number;
  key_scale?: string;
  vocal_language: string;
  song_model: string;
  audio_backend: "mlx" | "mps_torch";
  // Mode-specific:
  audio_cover_strength?: number;
  cover_noise_strength?: number;
  repainting_start?: number;
  repainting_end?: number;
  repaint_mode?: string;
  repaint_strength?: number;
  track_names?: string[];
  global_caption?: string;
  use_lora: boolean;
  lora_adapter_path: string;
  lora_adapter_name: string;
  use_lora_trigger: boolean;
  lora_trigger_tag: string;
  lora_scale: number;
  adapter_model_variant: string;
  adapter_song_model: string;
}

export interface SourceAudioWizardConfig {
  mode: WizardMode;
  title: string;
  subtitle: string;
  examples: string[];
  // Which mode-specific section to show on step 2
  variant: "cover" | "repaint" | "extract" | "lego" | "complete";
  defaultModel?: string;
}

export function SourceAudioWizard({ config }: { config: SourceAudioWizardConfig }) {
  const navigate = useNavigate();
  const setResult = useWizardStore((s) => s.setResult);
  const lastResult = useWizardStore((s) => s.lastResult[config.mode]);
  const warnings = useWizardStore((s) => s.warnings[config.mode]) ?? [];
  const draft = useWizardStore((s) => s.drafts[config.mode]);
  const sourceDefaults = React.useMemo<BaseSourceForm>(
    () => ({
      task_type: TASK_TYPE_BY_MODE[config.variant] ?? "cover",
      title: "",
      artist_name: "",
      caption: "",
      style_profile: "auto",
      tags: "",
      negative_tags: "",
      lyrics: config.variant === "extract" ? "[Instrumental]" : "",
      instrumental: config.variant === "extract",
      duration: 60,
      vocal_language: "en",
      song_model: config.defaultModel ?? "acestep-v15-xl-sft",
      audio_backend: "mlx",
      audio_cover_strength: 0.6,
      cover_noise_strength: 0.2,
      repainting_start: 0,
      repainting_end: 30,
      repaint_mode: "balanced",
      repaint_strength: 0.6,
      track_names: ["vocals", "drums", "bass"],
      global_caption: "",
      use_lora: false,
      lora_adapter_path: "",
      lora_adapter_name: "",
      use_lora_trigger: false,
      lora_trigger_tag: "",
      lora_scale: DEFAULT_LORA_SCALE,
      adapter_model_variant: "",
      adapter_song_model: "",
    }),
    [config.defaultModel, config.variant],
  );

  const form = useForm<BaseSourceForm>({
    defaultValues: mergeWizardDraft<BaseSourceForm>(sourceDefaults, draft),
    mode: "onChange",
  });

  const [step, setStep] = React.useState(0);
  const [aiPromptPending, setAiPromptPending] = React.useState(false);
  const [source, setSource] = React.useState<SourceAudioValue | undefined>();
  const [audioCodes, setAudioCodes] = React.useState<string>("");
  const storePrompt = useWizardStore((s) => s.prompts[config.mode]);
  const aiDescription = storePrompt ?? "";
  const values = form.watch();
  const draftState = useWizardDraft(config.mode, form);
  const baseOnlyModelError =
    BASE_ONLY_VARIANTS.has(config.variant) && !String(values.song_model || "").includes("-base")
      ? "Extract, Lego en Complete zijn ACE-Step Base-only taken. Kies ACE-Step v1.5 XL Base voordat je genereert."
      : "";

  React.useEffect(() => {
    if (source?.duration && (config.variant === "cover" || config.variant === "extract")) {
      form.setValue("duration", source.duration);
    }
    if (source?.duration && config.variant === "repaint") {
      form.setValue("repainting_end", Math.min(30, source.duration));
    }
  }, [source?.duration, config.variant, form]);

  // Auto-extract audio codes for lego/complete
  React.useEffect(() => {
    if (!source?.uploadId) return;
    if (config.variant !== "lego" && config.variant !== "complete") return;
    if (audioCodes) return;
    api
      .post<{ success: boolean; audio_code_string?: string; error?: string }>(
        "/api/audio-codes",
        { upload_id: source.uploadId },
      )
      .then((resp) => {
        if (resp.success && resp.audio_code_string) {
          setAudioCodes(resp.audio_code_string);
        } else if (resp.error) {
          toast.error(`Audio-codes: ${resp.error}`);
        }
      })
      .catch((e) => toast.error(`Audio-codes: ${(e as Error).message}`));
  }, [source?.uploadId, config.variant, audioCodes]);

  const hydrate = (payload: Record<string, unknown>) => {
    for (const k of Object.keys(form.getValues())) {
      if (k in payload) {
        // @ts-expect-error dynamic
        form.setValue(k, payload[k]);
      }
    }
    if (Array.isArray(payload.track_names)) {
      form.setValue("track_names", payload.track_names as string[]);
    }
    draftState.saveNow(form.getValues());
  };

  const generation = useGenerationJobRunner({
    mode: config.mode,
    label: config.title,
    onComplete: (resp) => {
      setResult(config.mode, resp as unknown as Record<string, unknown>);
      setStep(999);
    },
  });

  const setLoraSelection = (selection: LoraSelection) => {
    form.setValue("use_lora", selection.use_lora, { shouldValidate: true });
    form.setValue("lora_adapter_path", selection.lora_adapter_path, { shouldValidate: true });
    form.setValue("lora_adapter_name", selection.lora_adapter_name, { shouldValidate: true });
    form.setValue("use_lora_trigger", selection.use_lora_trigger, { shouldValidate: true });
    form.setValue("lora_trigger_tag", selection.lora_trigger_tag, { shouldValidate: true });
    form.setValue("lora_scale", selection.lora_scale, { shouldValidate: true });
    form.setValue("adapter_model_variant", selection.adapter_model_variant, { shouldValidate: true });
    form.setValue("adapter_song_model", selection.adapter_song_model, { shouldValidate: true });
    if (selection.use_lora && selection.adapter_song_model) {
      form.setValue("song_model", selection.adapter_song_model, { shouldValidate: true });
    }
  };

  const buildPayload = () => {
    const v = form.getValues();
    const payload: Record<string, unknown> = {
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
      vocal_language: v.vocal_language,
      song_model: v.song_model,
      audio_backend: v.audio_backend,
      use_mlx_dit: useMlxDitForAudioBackend(v.audio_backend),
      src_audio_id: source?.uploadId,
      ...normalizeLoraSelection(v),
    };
    if (config.variant === "cover") {
      payload.audio_cover_strength = v.audio_cover_strength;
      payload.cover_noise_strength = v.cover_noise_strength;
    }
    if (config.variant === "repaint") {
      payload.repainting_start = v.repainting_start;
      payload.repainting_end = v.repainting_end;
      payload.repaint_mode = v.repaint_mode;
      payload.repaint_strength = v.repaint_strength;
    }
    if (config.variant === "extract" || config.variant === "lego" || config.variant === "complete") {
      payload.track_names = v.track_names;
    }
    if (config.variant === "lego" || config.variant === "complete") {
      payload.audio_code_string = audioCodes;
      payload.global_caption = v.global_caption;
    }
    return payload;
  };

  const audioUrl =
    firstGenerationAudioUrl(lastResult) ||
    (typeof lastResult?.audio === "string"
      ? `data:audio/wav;base64,${lastResult.audio}`
      : undefined);

  const renderModeStep = () => {
    if (config.variant === "cover") {
      return (
        <FieldGroup
          title="Cover-sterkte"
          description="Hoe sterk vasthouden aan het origineel (1) of geheel hercomponeren (0)."
        >
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-3">
              <div className="flex items-baseline justify-between">
                <Label>Cover strength</Label>
                <span className="font-mono text-xs">{(values.audio_cover_strength ?? 0).toFixed(2)}</span>
              </div>
              <Controller
                control={form.control}
                name="audio_cover_strength"
                render={({ field }) => (
                  <Slider value={[field.value ?? 0.6]} min={0} max={1} step={0.01} onValueChange={(v) => field.onChange(v[0])} />
                )}
              />
            </div>
            <div className="space-y-3">
              <div className="flex items-baseline justify-between">
                <Label>Noise strength</Label>
                <span className="font-mono text-xs">{(values.cover_noise_strength ?? 0).toFixed(2)}</span>
              </div>
              <Controller
                control={form.control}
                name="cover_noise_strength"
                render={({ field }) => (
                  <Slider value={[field.value ?? 0.2]} min={0} max={1} step={0.01} onValueChange={(v) => field.onChange(v[0])} />
                )}
              />
            </div>
          </div>
        </FieldGroup>
      );
    }
    if (config.variant === "repaint") {
      const max = source?.duration ?? 600;
      return (
        <FieldGroup
          title="Repaint regio"
          description="Welk deel van de track wordt opnieuw geschilderd."
        >
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-3">
              <div className="flex items-baseline justify-between">
                <Label>Start</Label>
                <span className="font-mono text-xs">{formatDuration(values.repainting_start ?? 0)}</span>
              </div>
              <Controller
                control={form.control}
                name="repainting_start"
                render={({ field }) => (
                  <Slider value={[field.value ?? 0]} min={0} max={max} step={1} onValueChange={(v) => field.onChange(v[0])} />
                )}
              />
            </div>
            <div className="space-y-3">
              <div className="flex items-baseline justify-between">
                <Label>End</Label>
                <span className="font-mono text-xs">{formatDuration(values.repainting_end ?? 30)}</span>
              </div>
              <Controller
                control={form.control}
                name="repainting_end"
                render={({ field }) => (
                  <Slider value={[field.value ?? 30]} min={0} max={max} step={1} onValueChange={(v) => field.onChange(v[0])} />
                )}
              />
            </div>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label>Mode</Label>
              <Controller
                control={form.control}
                name="repaint_mode"
                render={({ field }) => (
                  <Select value={field.value} onValueChange={field.onChange}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="balanced">Balanced</SelectItem>
                      <SelectItem value="strength">Strength-based</SelectItem>
                    </SelectContent>
                  </Select>
                )}
              />
            </div>
            <div className="space-y-3">
              <div className="flex items-baseline justify-between">
                <Label>Strength</Label>
                <span className="font-mono text-xs">{(values.repaint_strength ?? 0.6).toFixed(2)}</span>
              </div>
              <Controller
                control={form.control}
                name="repaint_strength"
                render={({ field }) => (
                  <Slider value={[field.value ?? 0.6]} min={0} max={1} step={0.01} onValueChange={(v) => field.onChange(v[0])} />
                )}
              />
            </div>
          </div>
        </FieldGroup>
      );
    }
    if (config.variant === "extract" || config.variant === "lego" || config.variant === "complete") {
      return (
        <FieldGroup
          title="Stems"
          description="Welke stems wil je isoleren / regenereren?"
        >
          <div className="flex flex-wrap gap-2">
            {STEM_OPTIONS.map((s) => {
              const active = (values.track_names ?? []).includes(s);
              return (
                <button
                  key={s}
                  type="button"
                  onClick={() => {
                    const next = active
                      ? (values.track_names ?? []).filter((x) => x !== s)
                      : [...(values.track_names ?? []), s];
                    form.setValue("track_names", next);
                  }}
                  className={cn(
                    "rounded-full border px-3 py-1 text-xs transition-colors",
                    active
                      ? "border-primary/60 bg-primary/15 text-primary"
                      : "border-border/60 text-muted-foreground hover:border-foreground/40",
                  )}
                >
                  {s}
                </button>
              );
            })}
          </div>
          {(config.variant === "lego" || config.variant === "complete") && (
            <div className="space-y-1.5">
              <Label>Global caption</Label>
              <Textarea rows={2} {...form.register("global_caption")} placeholder="Sfeer / vibe over alle stems heen" />
              {audioCodes && (
                <Badge variant="muted" className="text-[10px]">
                  Audio codes geladen ({audioCodes.length} bytes)
                </Badge>
              )}
            </div>
          )}
        </FieldGroup>
      );
    }
    return null;
  };

  const steps: WizardStepDef[] = [
    {
      key: "ai",
      title: "AI prompt",
      description:
        "Beschrijf wat je met de bron-audio wilt doen. AI vult tags, lyrics en parameters in.",
      isValid: (aiDescription.trim().length >= 4 || !!source) && !aiPromptPending,
      render: () => (
        <AIPromptStep
          mode={config.mode}
          placeholder={config.examples[0]}
          examples={config.examples}
          onPendingChange={setAiPromptPending}
          onHydrated={(payload) => {
            hydrate(payload);
            draftState.saveNow(form.getValues());
          }}
        />
      ),
    },
    {
      key: "source",
      title: "Bron-audio",
      description:
        "Upload de WAV/MP3 die als referentie dient. Wordt automatisch geconverteerd naar ACE-Step codes waar nodig.",
      isValid: !!source?.uploadId,
      render: () => (
        <SourceAudioStep value={source} onChange={setSource} />
      ),
    },
    {
      key: "mode",
      title: `${config.title}-instellingen`,
      isValid: true,
      render: () => (
        <div className="space-y-4">
          <FieldGroup title="Model & backend">
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1.5">
                <Label>Song model</Label>
                <Controller
                  control={form.control}
                  name="song_model"
                  render={({ field }) => (
                    <Select value={field.value} onValueChange={field.onChange}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {SONG_MODELS.map(([id, label]) => (
                          <SelectItem key={id} value={id}>{label}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                />
                {baseOnlyModelError && (
                  <p className="text-xs text-amber-500">{baseOnlyModelError}</p>
                )}
              </div>
              <AudioBackendSelector
                value={values.audio_backend}
                onChange={(value) => form.setValue("audio_backend", value, { shouldValidate: true })}
              />
            </div>
          </FieldGroup>
          {renderModeStep()}
        </div>
      ),
    },
    {
      key: "lora",
      title: "LoRA",
      description: "Optioneel: kies een generation-loadable PEFT LoRA adapter.",
      isValid: true,
      render: () => (
        <FieldGroup title="LoRA adapter">
          <LoraSelector value={values} onChange={setLoraSelection} />
        </FieldGroup>
      ),
    },
    {
      key: "fields",
      title: "Identiteit & sfeer",
      isValid: true,
      hidden: config.variant === "extract",
      render: () => (
        <div className="space-y-4">
          <FieldGroup title="Naam">
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1.5">
                <Label>Titel</Label>
                <Input {...form.register("title")} />
              </div>
              <div className="space-y-1.5">
                <Label>Artiest</Label>
                <Input {...form.register("artist_name")} />
              </div>
            </div>
          </FieldGroup>
          <FieldGroup title="Caption & tags">
            <AudioStyleSelector
              value={values.style_profile}
              onChange={(value) => form.setValue("style_profile", value, { shouldValidate: true })}
            />
            <div className="space-y-1.5">
              <Label>Caption</Label>
              <Textarea rows={2} {...form.register("caption")} />
            </div>
            <div className="space-y-1.5">
              <Label>Tags</Label>
              <Controller
                control={form.control}
                name="tags"
                render={({ field }) => (
                  <TagInput value={field.value} onChange={field.onChange} />
                )}
              />
            </div>
            <div className="space-y-1.5">
              <Label>Negative tags</Label>
              <Controller
                control={form.control}
                name="negative_tags"
                render={({ field }) => (
                  <TagInput value={field.value} onChange={field.onChange} variant="negative" />
                )}
              />
            </div>
          </FieldGroup>
          {config.variant !== "extract" && (
            <FieldGroup title="Lyrics & taal">
              <div className="grid gap-3 sm:grid-cols-[200px_1fr]">
                <div className="space-y-1.5">
                  <Label>Taal</Label>
                  <Input {...form.register("vocal_language")} />
                </div>
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between">
                    <Label>Instrumentaal</Label>
                    <Controller
                      control={form.control}
                      name="instrumental"
                      render={({ field }) => (
                        <Switch checked={field.value} onCheckedChange={field.onChange} />
                      )}
                    />
                  </div>
                </div>
              </div>
              {!values.instrumental && (
                <div className="space-y-1.5">
                  <Label>Lyrics</Label>
                  <Textarea rows={8} className="font-mono text-xs" {...form.register("lyrics")} />
                </div>
              )}
            </FieldGroup>
          )}
        </div>
      ),
    },
    {
      key: "review",
      title: "Review & genereer",
      isValid: !!source?.uploadId && !baseOnlyModelError,
      render: () => (
        <div className="space-y-4">
          {baseOnlyModelError && (
            <FieldGroup title="Model blokkade" description={baseOnlyModelError}>
              <p className="text-sm text-muted-foreground">
                Ga terug naar de modelstap en kies ACE-Step v1.5 XL Base om deze source-audio taak te renderen.
              </p>
            </FieldGroup>
          )}
          <ReviewStep
            payload={buildPayload()}
            warnings={warnings}
            primaryFields={[
              { key: "task_type", label: "Modus" },
              { key: "title", label: "Titel" },
              { key: "song_model", label: "Model" },
              { key: "audio_backend", label: "Backend", format: audioBackendLabel },
              { key: "lora_adapter_name", label: "LoRA" },
              { key: "lora_trigger_tag", label: "LoRA trigger" },
              { key: "duration", label: "Duur", format: (v) => formatDuration(Number(v) || 0) },
              { key: "vocal_language", label: "Taal" },
              { key: "src_audio_id", label: "Source ID" },
            ]}
          />
          <RenderInsightPanel payload={buildPayload()} warnings={warnings} />
          {(generation.jobId || generation.isSubmitting) && (
            <GenerationJobStatus
              job={generation.activeJob}
              jobId={generation.jobId}
              onOpen={generation.openActiveJob}
            />
          )}
        </div>
      ),
    },
    {
      key: "result",
      title: "Resultaat",
      isValid: true,
      hidden: !lastResult,
      render: () => {
        if (!lastResult) return null;
        const resultId =
          (lastResult.result_id as string | undefined) ||
          (lastResult.id as string | undefined);
        const title =
          (lastResult.title as string | undefined) || values.title || config.title;
        const artist =
          (lastResult.artist_name as string | undefined) || values.artist_name;
        return (
          <div className="space-y-4">
            <GenerationAudioList result={lastResult} title={title} artist={artist} />
            <motion.div
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex flex-wrap gap-2"
            >
              <Button variant="outline" onClick={() => navigate("/library")} className="gap-2">
                <Music4 className="size-4" /> Open library
              </Button>
              {audioUrl && (
                <Button
                  variant="outline"
                  onClick={() => navigate("/wizard/video", {
                    state: {
                      audio_url: audioUrl,
                      title,
                      artist_name: artist,
                      prompt: String(lastResult.caption || values.caption || values.tags || title || ""),
                      target_type: "song",
                      target_id: resultId || title,
                    },
                  })}
                  className="gap-2"
                >
                  <Video className="size-4" /> Create video
                </Button>
              )}
              <Button variant="ghost" onClick={() => { form.reset(sourceDefaults); draftState.clear(); setStep(0); }}>Nieuw {config.title.toLowerCase()}</Button>
            </motion.div>
          </div>
        );
      },
    },
  ];

  return (
    <WizardShell
      title={config.title}
      subtitle={config.subtitle}
      steps={steps}
      step={step}
      onStepChange={setStep}
      onFinish={() => void generation.start(buildPayload())}
      isFinishing={generation.isSubmitting || generation.isRunning}
      finishLabel={generation.isSubmitting || generation.isRunning ? "Renderen…" : "Genereer"}
    />
  );
}
