import * as React from "react";
import { useNavigate } from "react-router-dom";
import { useForm, Controller } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { motion } from "framer-motion";
import { Music4, ImagePlus, Video } from "lucide-react";

import { WizardShell, FieldGroup, type WizardStepDef } from "@/components/wizard/WizardShell";
import { AIPromptStep } from "@/components/wizard/AIPromptStep";
import { ReviewStep } from "@/components/wizard/ReviewStep";
import { GenerationJobStatus } from "@/components/wizard/GenerationJobStatus";
import { GenerationAudioList, firstGenerationAudioUrl } from "@/components/wizard/GenerationAudioList";
import { LoraSelector } from "@/components/wizard/LoraSelector";
import { RenderInsightPanel } from "@/components/wizard/RenderInsightPanel";
import { QualityPresets } from "@/components/wizard/QualityPresets";
import { AutomationFields } from "@/components/wizard/AutomationFields";
import { AudioStyleSelector } from "@/components/wizard/AudioStyleSelector";
import { AudioBackendSelector } from "@/components/wizard/AudioBackendSelector";
import { MfluxArtMaker } from "@/components/mflux/MfluxArtMaker";
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
import { simpleSchema, simpleDefaults, type SimpleFormValues } from "@/lib/schemas";
import { ACE_STEP_KEY_SCALE_OPTIONS, ACE_STEP_TIME_SIGNATURE_OPTIONS } from "@/lib/aceStepSettings";
import { ACE_STEP_LANGUAGE_OPTIONS } from "@/lib/languages";
import { normalizeLoraSelection, type LoraSelection } from "@/lib/lora";
import { audioBackendLabel, useMlxDitForAudioBackend } from "@/lib/audioBackend";
import { useGenerationJobRunner } from "@/hooks/useGenerationJobRunner";
import { mergeWizardDraft, usePromptMirror, useWizardDraft } from "@/hooks/useWizardDraft";
import { useWizardStore } from "@/store/wizard";
import { formatDuration } from "@/lib/utils";

const MODE = "simple" as const;

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

export function SimpleWizard() {
  const navigate = useNavigate();
  const setResult = useWizardStore((s) => s.setResult);
  const lastResult = useWizardStore((s) => s.lastResult[MODE]);
  const warnings = useWizardStore((s) => s.warnings[MODE]) ?? [];
  const storePrompt = useWizardStore((s) => s.prompts[MODE]);
  const draft = useWizardStore((s) => s.drafts[MODE]);

  const form = useForm<SimpleFormValues>({
    resolver: zodResolver(simpleSchema),
    defaultValues: mergeWizardDraft<SimpleFormValues>(simpleDefaults, draft),
    mode: "onChange",
  });

  const [step, setStep] = React.useState(0);
  const [aiPromptPending, setAiPromptPending] = React.useState(false);
  const values = form.watch();
  const draftState = useWizardDraft(MODE, form);

  usePromptMirror(form, "simple_description", storePrompt);

  const generation = useGenerationJobRunner({
    mode: MODE,
    label: "track",
    onComplete: (resp) => {
      setResult(MODE, resp as unknown as Record<string, unknown>);
      setStep(4);
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

  const hydrate = (payload: Record<string, unknown>) => {
    const next: Partial<SimpleFormValues> = {};
    for (const [k, v] of Object.entries(payload)) {
      if (k in simpleDefaults || k === "simple_description") {
        // @ts-expect-error dynamic key set
        next[k] = v;
      }
    }
    const merged = { ...form.getValues(), ...next };
    form.reset(merged);
    draftState.saveNow(merged);
    return merged;
  };

  const buildPayload = () => {
    const v = form.getValues();
    return {
      task_type: "text2music",
      simple_description: v.simple_description,
      song_description: v.simple_description,
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
      auto_song_art: v.auto_song_art,
      auto_album_art: false,
      auto_video_clip: v.auto_video_clip,
      art_prompt: v.art_prompt,
      video_prompt: v.video_prompt,
      ...normalizeLoraSelection(v),
    };
  };

  const handleFinish = () => {
    void generation.start(buildPayload());
  };

  const audioUrl =
    firstGenerationAudioUrl(lastResult) ||
    (typeof lastResult?.audio === "string"
      ? `data:audio/wav;base64,${lastResult.audio}`
      : undefined);

  const steps: WizardStepDef[] = [
    {
      key: "ai",
      title: "AI prompt",
      description:
        "Beschrijf je idee in eigen woorden — de AI vult de rest van de wizard alvast in.",
      isValid: (values.simple_description ?? "").trim().length >= 4 && !aiPromptPending,
      render: () => (
        <AIPromptStep
          mode="simple"
          placeholder="Bijv. 'energetic synthwave instrumental, 90s sci-fi feel, neon pads, 110 bpm'"
          examples={[
            "uplifting indie-pop ballad, female vocals, 80 bpm",
            "lo-fi hip-hop instrumental with rain sfx, mellow",
            "cinematic orchestral hybrid, 6/8, building to drop",
          ]}
          onPendingChange={setAiPromptPending}
          onHydrated={(payload) => {
            const merged = hydrate(payload);
            // Sync AI prompt back into the form so isValid passes
            const desc =
              (payload.simple_description as string | undefined) ??
              (payload.song_description as string | undefined) ??
              form.getValues("simple_description");
            if (desc) {
              const withDesc = { ...merged, simple_description: desc };
              form.reset(withDesc);
              draftState.saveNow(withDesc);
            }
          }}
        />
      ),
    },
    {
      key: "identity",
      title: "Identiteit",
      description: "Titel, artiest en taal — voorgevuld door AI, hier nog aanpasbaar.",
      isValid: true,
      render: () => (
        <div className="space-y-4">
          <FieldGroup title="Titel & artiest">
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1.5">
                <Label htmlFor="title">Titel</Label>
                <Input id="title" placeholder="Bijv. Neon Horizon" {...form.register("title")} />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="artist">Artiest</Label>
                <Input id="artist" placeholder="Bijv. Aurora Kite" {...form.register("artist_name")} />
              </div>
            </div>
          </FieldGroup>

          <FieldGroup title="Taal & lyrics">
            <div className="grid gap-3 sm:grid-cols-[200px_1fr]">
              <div className="space-y-1.5">
                <Label>Taal</Label>
                <Controller
                  control={form.control}
                  name="vocal_language"
                  render={({ field }) => (
                    <Select value={field.value} onValueChange={field.onChange}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {ACE_STEP_LANGUAGE_OPTIONS.map(([code, name]) => (
                          <SelectItem key={code} value={code}>{name}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                />
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
                <p className="text-xs text-muted-foreground">
                  Aan = geen vocals, lyrics-veld wordt genegeerd.
                </p>
              </div>
            </div>
            {!values.instrumental && (
              <div className="space-y-1.5">
                <Label htmlFor="lyrics">Lyrics</Label>
                <Textarea
                  id="lyrics"
                  rows={8}
                  className="font-mono text-xs"
                  placeholder={"[Verse 1]\n…\n\n[Chorus]\n…"}
                  {...form.register("lyrics")}
                />
              </div>
            )}
          </FieldGroup>

          <FieldGroup title="Tags & sfeer" description="Komma-gescheiden lijst — gebruik genres, instrumenten, mood-tags.">
            <div className="grid gap-3">
              <AudioStyleSelector
                value={values.style_profile}
                onChange={(value) => form.setValue("style_profile", value, { shouldValidate: true })}
              />
              <div className="space-y-1.5">
                <Label htmlFor="caption">Caption</Label>
                <Textarea id="caption" rows={2} {...form.register("caption")} />
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="space-y-1.5">
                  <Label htmlFor="tags">Tags</Label>
                  <Textarea id="tags" rows={2} {...form.register("tags")} />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="neg">Negative tags</Label>
                  <Textarea id="neg" rows={2} placeholder="Bijv. muddy mix, generic lyrics" {...form.register("negative_tags")} />
                </div>
              </div>
            </div>
          </FieldGroup>
        </div>
      ),
    },
    {
      key: "render",
      title: "Render",
      description: "Lengte, model, kwaliteit, optioneel toonsoort en bpm.",
      isValid: true,
      render: () => (
        <div className="space-y-4">
          <FieldGroup title="Lengte">
            <div className="space-y-3">
              <div className="flex items-baseline justify-between">
                <Label>Duur</Label>
                <span className="font-mono text-sm">{formatDuration(values.duration)}</span>
              </div>
              <Controller
                control={form.control}
                name="duration"
                render={({ field }) => (
                  <Slider
                    value={[field.value ?? 60]}
                    min={20}
                    max={600}
                    step={5}
                    onValueChange={(v) => field.onChange(v[0] ?? 60)}
                  />
                )}
              />
            </div>
          </FieldGroup>

          <FieldGroup title="Kwaliteit" description="Kies een preset; je kunt onder 'Muzikaal' nog fine-tunen.">
            <QualityPresets
              value={values.quality_profile}
              onChange={(p) => form.setValue("quality_profile", p)}
            />
          </FieldGroup>
          <FieldGroup title="Song model">
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
          </FieldGroup>

          <FieldGroup title="Audio backend">
            <AudioBackendSelector
              value={values.audio_backend}
              onChange={(value) => form.setValue("audio_backend", value, { shouldValidate: true })}
            />
          </FieldGroup>

          <FieldGroup title="LoRA" description="Optioneel: pas een getrainde PEFT LoRA toe op deze render.">
            <LoraSelector value={values} onChange={setLoraSelection} />
          </FieldGroup>

          <AutomationFields control={form.control} register={form.register} values={values} />

          <FieldGroup title="Muzikaal (optioneel)" description="Laat leeg om door het model te laten kiezen.">
            <div className="grid gap-3 sm:grid-cols-3">
              <div className="space-y-1.5">
                <Label>BPM</Label>
                <Input
                  type="number"
                  min={40}
                  max={300}
                  placeholder="auto"
                  {...form.register("bpm", { valueAsNumber: true })}
                />
              </div>
              <div className="space-y-1.5">
                <Label>Key</Label>
                <Controller
                  control={form.control}
                  name="key_scale"
                  render={({ field }) => (
                    <Select
                      value={field.value || "auto"}
                      onValueChange={(value) => field.onChange(value === "auto" ? undefined : value)}
                    >
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {ACE_STEP_KEY_SCALE_OPTIONS.map((value) => (
                          <SelectItem key={value} value={value}>{value === "auto" ? "Auto" : value}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>
              <div className="space-y-1.5">
                <Label>Time signature</Label>
                <Controller
                  control={form.control}
                  name="time_signature"
                  render={({ field }) => (
                    <Select
                      value={field.value || "auto"}
                      onValueChange={(value) => field.onChange(value === "auto" ? undefined : value)}
                    >
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {ACE_STEP_TIME_SIGNATURE_OPTIONS.map(([value, label]) => (
                          <SelectItem key={value || "auto"} value={value || "auto"}>{label}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>
            </div>
            <div className="space-y-1.5">
              <Label>Seed</Label>
              <Input
                type="number"
                placeholder="-1 voor random"
                {...form.register("seed", { valueAsNumber: true })}
              />
            </div>
          </FieldGroup>
        </div>
      ),
    },
    {
      key: "review",
      title: "Review & genereer",
      description: "Controleer de payload, en klik op Genereer.",
      isValid: form.formState.isValid,
      render: () => (
        <div className="space-y-4">
          <ReviewStep
            payload={buildPayload()}
            warnings={warnings}
            primaryFields={[
              { key: "title", label: "Titel" },
              { key: "artist_name", label: "Artiest" },
              { key: "duration", label: "Duur", format: (v) => formatDuration(Number(v) || 0) },
              { key: "song_model", label: "Model" },
              { key: "audio_backend", label: "Backend", format: audioBackendLabel },
              { key: "lora_adapter_name", label: "LoRA" },
              { key: "lora_trigger_tag", label: "LoRA trigger" },
              { key: "vocal_language", label: "Taal" },
              { key: "quality_profile", label: "Kwaliteit" },
              { key: "tags", label: "Tags" },
              { key: "bpm", label: "BPM" },
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
      description: "Speel je track af, genereer artwork, of bouw verder.",
      isValid: true,
      hidden: !lastResult,
      render: () => {
        if (!lastResult) return null;
        const resultId =
          (lastResult.result_id as string | undefined) ||
          (lastResult.song_id as string | undefined) ||
          (lastResult.id as string | undefined);
        const title = (lastResult.title as string | undefined) || values.title;
        const artist = (lastResult.artist_name as string | undefined) || values.artist_name;
        const sourceResultId = lastResult.result_id as string | undefined;
        const songId = lastResult.song_id as string | undefined;
        return (
          <div className="space-y-4">
            <GenerationAudioList result={lastResult} title={title} artist={artist} />
            <MfluxArtMaker
              title={title}
              artist={artist}
              context={String(lastResult.caption || values.caption || values.tags || "")}
              targetType={sourceResultId ? "generation_result" : "song"}
              targetId={sourceResultId || songId}
            />
            <motion.div
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex flex-wrap items-center gap-2"
            >
              {Array.isArray(lastResult.payload_warnings) &&
                (lastResult.payload_warnings as string[]).map((w, i) => (
                  <Badge key={i} variant="muted" className="text-[11px]">
                    {w}
                  </Badge>
                ))}
              <Button variant="outline" onClick={() => navigate("/library")} className="gap-2">
                <Music4 className="size-4" />
                Open library
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
                      target_id: songId || sourceResultId || title,
                    },
                  })}
                  className="gap-2"
                >
                  <Video className="size-4" />
                  Create video
                </Button>
              )}
              <Button variant="ghost" onClick={() => { form.reset(simpleDefaults); draftState.clear(); setStep(0); }} className="gap-2">
                <ImagePlus className="size-4" />
                Nog één maken
              </Button>
            </motion.div>
          </div>
        );
      },
    },
  ];

  return (
    <WizardShell
      title="Simple wizard"
      subtitle="Een track in 4 stappen: AI vult voor, jij kiest, ACE-Step rendert."
      steps={steps}
      step={step}
      onStepChange={setStep}
      onFinish={handleFinish}
      isFinishing={generation.isSubmitting || generation.isRunning}
      finishLabel={generation.isSubmitting || generation.isRunning ? "Rendert…" : "Genereer track"}
    />
  );
}
