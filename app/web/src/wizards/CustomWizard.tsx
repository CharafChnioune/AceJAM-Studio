import * as React from "react";
import { useNavigate } from "react-router-dom";
import { useForm, Controller } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { motion } from "framer-motion";
import { Music4, Video } from "lucide-react";

import { WizardShell, FieldGroup, type WizardStepDef } from "@/components/wizard/WizardShell";
import { AIPromptStep } from "@/components/wizard/AIPromptStep";
import { ReviewStep } from "@/components/wizard/ReviewStep";
import { GenerationJobStatus } from "@/components/wizard/GenerationJobStatus";
import { GenerationAudioList, firstGenerationAudioUrl } from "@/components/wizard/GenerationAudioList";
import { LoraSelector } from "@/components/wizard/LoraSelector";
import { RenderInsightPanel } from "@/components/wizard/RenderInsightPanel";
import { AutomationFields } from "@/components/wizard/AutomationFields";
import { TagInput } from "@/components/wizard/TagInput";
import { AudioStyleSelector } from "@/components/wizard/AudioStyleSelector";
import { AudioBackendSelector } from "@/components/wizard/AudioBackendSelector";
import { AceStepAdvancedSettings } from "@/components/wizard/AceStepAdvancedSettings";
import { MfluxArtMaker } from "@/components/mflux/MfluxArtMaker";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
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
import { customSchema, simpleDefaults, type CustomFormValues } from "@/lib/schemas";
import {
  ACE_STEP_ADVANCED_DEFAULTS,
  ACE_STEP_ADVANCED_PAYLOAD_FIELDS,
  ACE_STEP_KEY_SCALE_OPTIONS,
  ACE_STEP_TIME_SIGNATURE_OPTIONS,
  OFFICIAL_AUDIO_FORMAT_OPTIONS,
} from "@/lib/aceStepSettings";
import { ACE_STEP_LANGUAGE_OPTIONS } from "@/lib/languages";
import { normalizeLoraSelection, type LoraSelection } from "@/lib/lora";
import { audioBackendLabel, useMlxDitForAudioBackend } from "@/lib/audioBackend";
import { useGenerationJobRunner } from "@/hooks/useGenerationJobRunner";
import { mergeWizardDraft, usePromptMirror, useWizardDraft } from "@/hooks/useWizardDraft";
import { useWizardStore } from "@/store/wizard";
import { formatDuration } from "@/lib/utils";

const MODE = "custom" as const;

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

function docsCorrectRenderDefaults(songModel: string) {
  if (songModel.includes("turbo")) {
    return { inference_steps: 8, shift: 3 };
  }
  return { inference_steps: 50, shift: 1 };
}

const TAG_SUGGESTIONS = [
  "synthwave", "lofi", "indie pop", "edm", "house", "techno", "ambient",
  "cinematic", "orchestral", "rock", "metal", "pop", "rap", "trap", "drill",
  "afrobeat", "reggaeton", "disco", "funk", "soul", "rnb", "jazz", "blues",
  "classical", "country", "folk", "punk", "drum & bass", "dubstep",
  "uplifting", "melancholic", "energetic", "chill", "dark", "epic",
  "kick", "808", "subbass", "lead synth", "pad", "piano", "guitar",
  "strings", "brass", "vocal chops", "side-chain", "reverb", "delay",
];

const NEG_SUGGESTIONS = [
  "muddy mix", "generic lyrics", "weak hook", "off-key vocals",
  "clipping", "low quality", "amateur production", "pitchy vocals",
];

export function CustomWizard() {
  const navigate = useNavigate();
  const setResult = useWizardStore((s) => s.setResult);
  const lastResult = useWizardStore((s) => s.lastResult[MODE]);
  const warnings = useWizardStore((s) => s.warnings[MODE]) ?? [];
  const storePrompt = useWizardStore((s) => s.prompts[MODE]);
  const draft = useWizardStore((s) => s.drafts[MODE]);
  const customDefaults = React.useMemo<CustomFormValues>(
    () => ({
      ...simpleDefaults,
      task_type: "text2music",
      inference_steps: 50,
      guidance_scale: 7,
      shift: 1,
      audio_format: "wav32",
      batch_size: 1,
      ...(ACE_STEP_ADVANCED_DEFAULTS as Partial<CustomFormValues>),
    }),
    [],
  );

  const form = useForm<CustomFormValues>({
    resolver: zodResolver(customSchema),
    defaultValues: mergeWizardDraft<CustomFormValues>(customDefaults, draft),
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
      setStep(7);
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
      const defaults = docsCorrectRenderDefaults(selection.adapter_song_model);
      form.setValue("song_model", selection.adapter_song_model, { shouldValidate: true });
      form.setValue("inference_steps", defaults.inference_steps, { shouldValidate: true });
      form.setValue("shift", defaults.shift, { shouldValidate: true });
    }
  };

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
        // @ts-expect-error dynamic
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
      ...advanced,
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
      description: "Beschrijf je vibe — AI vult titel, tags, lyrics, bpm, key en alles erbij.",
      isValid: (values.simple_description ?? "").trim().length >= 4 && !aiPromptPending,
      render: () => (
        <AIPromptStep
          mode="custom"
          placeholder="Bijv. 'cinematic post-rock build, slow tempo, female lead, melancholic strings'"
          examples={[
            "afrobeat ode to a coastal town, 105 bpm, optimistic",
            "dark trap drill instrumental in F# minor, 140 bpm",
            "8-bit chiptune adventure theme, 4/4, major key",
          ]}
          onPendingChange={setAiPromptPending}
          onHydrated={(payload) => {
            const merged = hydrate(payload);
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
      title: "Identiteit & taal",
      isValid: true,
      render: () => (
        <div className="space-y-4">
          <FieldGroup title="Naam">
            <div className="grid gap-3 sm:grid-cols-3">
              <div className="space-y-1.5">
                <Label htmlFor="title">Titel</Label>
                <Input id="title" {...form.register("title")} />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="artist">Artiest</Label>
                <Input id="artist" {...form.register("artist_name")} />
              </div>
            </div>
          </FieldGroup>
          <FieldGroup title="Taal & vocals">
            <div className="grid gap-3 sm:grid-cols-[200px_1fr]">
              <div className="space-y-1.5">
                <Label>Taal</Label>
                <Controller
                  control={form.control}
                  name="vocal_language"
                  render={({ field }) => (
                    <Select
                      value={field.value}
                      onValueChange={field.onChange}
                    >
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
                  Aan = lyrics worden vervangen door [Instrumental].
                </p>
              </div>
            </div>
          </FieldGroup>
        </div>
      ),
    },
    {
      key: "sound",
      title: "Sound & sfeer",
      description: "Tags, caption, negative tags. Tikken-Enter om een tag vast te leggen.",
      isValid: true,
      render: () => (
        <div className="space-y-4">
          <FieldGroup title="Caption" description="Korte prozabeschrijving van de sound.">
            <AudioStyleSelector
              value={values.style_profile}
              onChange={(value) => form.setValue("style_profile", value, { shouldValidate: true })}
            />
            <Textarea rows={2} {...form.register("caption")} />
          </FieldGroup>
          <FieldGroup
            title="Tags"
            description="12-24 tags die samen het ACE-Step prompt vormen."
          >
            <Controller
              control={form.control}
              name="tags"
              render={({ field }) => (
                <TagInput
                  value={field.value ?? ""}
                  onChange={field.onChange}
                  suggestions={TAG_SUGGESTIONS}
                  placeholder="bijv. synthwave, 110 bpm, neon pads…"
                />
              )}
            />
          </FieldGroup>
          <FieldGroup title="Negative tags" description="Wat moet er NIET in.">
            <Controller
              control={form.control}
              name="negative_tags"
              render={({ field }) => (
                <TagInput
                  value={field.value ?? ""}
                  onChange={field.onChange}
                  suggestions={NEG_SUGGESTIONS}
                  variant="negative"
                  placeholder="bijv. muddy mix, generic lyrics…"
                />
              )}
            />
          </FieldGroup>
        </div>
      ),
    },
    {
      key: "lyrics",
      title: "Lyrics",
      description: values.instrumental
        ? "Wordt overgeslagen — instrumental staat aan."
        : "Volledige tekst met section tags zoals [Verse 1], [Chorus], [Bridge].",
      isValid: true,
      hidden: values.instrumental,
      render: () => (
        <FieldGroup title="Lyrics">
          <Textarea
            rows={16}
            className="font-mono text-xs leading-relaxed"
            placeholder={"[Intro]\n…\n\n[Verse 1]\n…\n\n[Chorus]\n…"}
            {...form.register("lyrics")}
          />
          <p className="text-xs text-muted-foreground">
            Sectietags worden door ACE-Step herkend voor structuur. Houd het
            aantal woorden per sectie redelijk t.o.v. de duur.
          </p>
        </FieldGroup>
      ),
    },
    {
      key: "musical",
      title: "Muzikale parameters",
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
          <FieldGroup title="BPM, key & maatsoort">
            <div className="grid gap-3 sm:grid-cols-3">
              <div className="space-y-1.5">
                <Label>BPM</Label>
                <Input type="number" min={40} max={300} placeholder="auto" {...form.register("bpm", { valueAsNumber: true })} />
              </div>
              <div className="space-y-1.5">
                <Label>Toonsoort</Label>
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
                <Label>Maatsoort</Label>
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
          </FieldGroup>
        </div>
      ),
    },
    {
      key: "render",
      title: "Render-instellingen",
      description: "Model, kwaliteit, en geavanceerde inference-knoppen.",
      isValid: true,
      render: () => (
        <div className="space-y-4">
          <FieldGroup title="Model & kwaliteit">
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1.5">
                <Label>Song model</Label>
                <Controller
                  control={form.control}
                  name="song_model"
                  render={({ field }) => (
                    <Select
                      value={field.value}
                      onValueChange={(nextModel) => {
                        field.onChange(nextModel);
                        const defaults = docsCorrectRenderDefaults(nextModel);
                        form.setValue("inference_steps", defaults.inference_steps);
                        form.setValue("shift", defaults.shift);
                      }}
                    >
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {SONG_MODELS.map(([id, label]) => (
                          <SelectItem key={id} value={id}>{label}</SelectItem>
                        ))}
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
                    <Select
                      value={field.value}
                      onValueChange={(nextProfile) => {
                        field.onChange(nextProfile);
                        const defaults = docsCorrectRenderDefaults(values.song_model);
                        form.setValue("inference_steps", defaults.inference_steps);
                        form.setValue("shift", defaults.shift);
                      }}
                    >
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {QUALITY_PROFILES.map(([id, label]) => (
                          <SelectItem key={id} value={id}>{label}</SelectItem>
                        ))}
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
          <FieldGroup title="LoRA" description="Optioneel: kies een getrainde PEFT LoRA voor deze render.">
            <LoraSelector value={values} onChange={setLoraSelection} />
          </FieldGroup>
          <AutomationFields control={form.control} register={form.register} values={values} />
          <FieldGroup title="Inference (geavanceerd)" description="Laat staan voor presets — overschrijf alleen als je weet wat je doet.">
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-3">
                <div className="flex items-baseline justify-between">
                  <Label>Inference steps</Label>
                  <span className="font-mono text-xs">{values.inference_steps}</span>
                </div>
                <Controller
                  control={form.control}
                  name="inference_steps"
                  render={({ field }) => (
                    <Slider value={[field.value]} min={4} max={100} step={1} onValueChange={(v) => field.onChange(v[0])} />
                  )}
                />
              </div>
              <div className="space-y-3">
                <div className="flex items-baseline justify-between">
                  <Label>Guidance scale</Label>
                  <span className="font-mono text-xs">{values.guidance_scale.toFixed(1)}</span>
                </div>
                <Controller
                  control={form.control}
                  name="guidance_scale"
                  render={({ field }) => (
                    <Slider value={[field.value]} min={1} max={15} step={0.1} onValueChange={(v) => field.onChange(v[0])} />
                  )}
                />
              </div>
              <div className="space-y-3">
                <div className="flex items-baseline justify-between">
                  <Label>Shift</Label>
                  <span className="font-mono text-xs">{values.shift.toFixed(1)}</span>
                </div>
                <Controller
                  control={form.control}
                  name="shift"
                  render={({ field }) => (
                    <Slider value={[field.value]} min={0} max={10} step={0.1} onValueChange={(v) => field.onChange(v[0])} />
                  )}
                />
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
                        {OFFICIAL_AUDIO_FORMAT_OPTIONS.map(([value, label]) => (
                          <SelectItem key={value} value={value}>{label}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>
              <div className="space-y-1.5">
                <Label>Aantal takes</Label>
                <Controller
                  control={form.control}
                  name="batch_size"
                  render={({ field }) => (
                    <Slider value={[field.value]} min={1} max={8} step={1} onValueChange={(v) => field.onChange(v[0])} />
                  )}
                />
                <p className="text-xs text-muted-foreground">
                  {values.batch_size} take{values.batch_size === 1 ? "" : "s"} · XL-SFT/Base rendert takes een voor een op MPS.
                </p>
              </div>
              <div className="space-y-1.5">
                <Label>Seed</Label>
                <Input
                  type="number"
                  placeholder="-1 voor random"
                  {...form.register("seed", { valueAsNumber: true })}
                />
              </div>
            </div>
          </FieldGroup>
          <FieldGroup
            title="Official ACE-Step controls"
            description="Alle officiële extra velden uit de ACE-Step inference docs: DCW, CFG, output, retake, source/repaint en runtime."
          >
            <AceStepAdvancedSettings values={values} onChange={setAdvancedValue} />
          </FieldGroup>
        </div>
      ),
    },
    {
      key: "review",
      title: "Review & genereer",
      isValid: form.formState.isValid,
      render: () => (
        <div className="space-y-4">
          <ReviewStep
            payload={buildPayload()}
            warnings={warnings}
            primaryFields={[
              { key: "title", label: "Titel" },
              { key: "artist_name", label: "Artiest" },
              { key: "task_type", label: "Modus" },
              { key: "song_model", label: "Model" },
              { key: "audio_backend", label: "Backend", format: audioBackendLabel },
              { key: "lora_adapter_name", label: "LoRA" },
              { key: "lora_trigger_tag", label: "LoRA trigger" },
              { key: "duration", label: "Duur", format: (v) => formatDuration(Number(v) || 0) },
              { key: "bpm", label: "BPM" },
              { key: "key_scale", label: "Key" },
              { key: "vocal_language", label: "Taal" },
              { key: "quality_profile", label: "Kwaliteit" },
              { key: "tags", label: "Tags" },
              { key: "inference_steps", label: "Steps" },
              { key: "batch_size", label: "Takes" },
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
              <Button variant="ghost" onClick={() => { form.reset(customDefaults); draftState.clear(); setStep(0); }}>Nieuwe track</Button>
            </motion.div>
          </div>
        );
      },
    },
  ];

  return (
    <WizardShell
      title="Custom wizard"
      subtitle="Volledige controle: AI vult voor, jij stuurt elk veld bij."
      steps={steps}
      step={step}
      onStepChange={setStep}
      onFinish={handleFinish}
      isFinishing={generation.isSubmitting || generation.isRunning}
      finishLabel={generation.isSubmitting || generation.isRunning ? "Rendert…" : "Genereer track"}
    />
  );
}
