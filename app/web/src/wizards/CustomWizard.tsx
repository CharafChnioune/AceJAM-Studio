import * as React from "react";
import { useNavigate } from "react-router-dom";
import { useForm, Controller } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { motion } from "framer-motion";
import { Music4 } from "lucide-react";

import { WizardShell, FieldGroup, type WizardStepDef } from "@/components/wizard/WizardShell";
import { AIPromptStep } from "@/components/wizard/AIPromptStep";
import { ReviewStep } from "@/components/wizard/ReviewStep";
import { GenerationJobStatus } from "@/components/wizard/GenerationJobStatus";
import { LoraSelector } from "@/components/wizard/LoraSelector";
import { RenderInsightPanel } from "@/components/wizard/RenderInsightPanel";
import { TagInput } from "@/components/wizard/TagInput";
import { WaveformPlayer } from "@/components/audio/WaveformPlayer";
import { ArtGenerator } from "@/components/art/ArtGenerator";
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
import { customSchema, simpleDefaults, type CustomFormValues, defaultTrackArtPrompt } from "@/lib/schemas";
import { normalizeLoraSelection, type LoraSelection } from "@/lib/lora";
import { useGenerationJobRunner } from "@/hooks/useGenerationJobRunner";
import { useWizardStore } from "@/store/wizard";
import { formatDuration } from "@/lib/utils";

const MODE = "custom" as const;

const LANGUAGES = [
  ["en", "English"],
  ["nl", "Nederlands"],
  ["es", "Español"],
  ["fr", "Français"],
  ["de", "Deutsch"],
  ["pt", "Português"],
  ["it", "Italiano"],
  ["ja", "日本語"],
  ["ko", "한국어"],
  ["zh", "中文"],
  ["ar", "العربية"],
  ["hi", "हिन्दी"],
] as const;

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
  ["draft", "Draft (snel, ~8 stappen)"],
  ["standard", "Standard (runtime-verified, 8 stappen, shift 3)"],
  ["chart_master", "Chart Master (runtime-verified, 8 stappen, shift 3)"],
] as const;

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

  const form = useForm<CustomFormValues>({
    resolver: zodResolver(customSchema),
    defaultValues: {
      ...simpleDefaults,
      task_type: "text2music",
      inference_steps: 8,
      guidance_scale: 7,
      shift: 3,
      audio_format: "wav32",
      batch_size: 1,
    },
    mode: "onChange",
  });

  const [step, setStep] = React.useState(0);
  const [aiPromptPending, setAiPromptPending] = React.useState(false);
  const values = form.watch();

  React.useEffect(() => {
    if ((storePrompt ?? "") !== (form.getValues("simple_description") ?? "")) {
      form.setValue("simple_description", storePrompt ?? "", { shouldValidate: true });
    }
  }, [storePrompt, form]);

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
    form.setValue("lora_scale", selection.lora_scale, { shouldValidate: true });
    form.setValue("adapter_model_variant", selection.adapter_model_variant, { shouldValidate: true });
  };

  const hydrate = (payload: Record<string, unknown>) => {
    const next: Partial<CustomFormValues> = {};
    for (const [k, v] of Object.entries(payload)) {
      if (k in form.getValues() || k === "simple_description") {
        // @ts-expect-error dynamic
        next[k] = v;
      }
    }
    form.reset({ ...form.getValues(), ...next });
  };

  const buildPayload = () => {
    const v = form.getValues();
    return {
      task_type: v.task_type,
      title: v.title,
      artist_name: v.artist_name,
      caption: v.caption,
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
      quality_profile: v.quality_profile,
      seed: v.seed,
      inference_steps: v.inference_steps,
      guidance_scale: v.guidance_scale,
      shift: v.shift,
      audio_format: v.audio_format,
      batch_size: v.batch_size,
      ...normalizeLoraSelection(v),
    };
  };

  const handleFinish = () => {
    void generation.start(buildPayload());
  };

  const audioUrl =
    (lastResult?.audio_url as string | undefined) ||
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
            hydrate(payload);
            const desc =
              (payload.simple_description as string | undefined) ??
              (payload.song_description as string | undefined) ??
              form.getValues("simple_description");
            if (desc) form.setValue("simple_description", desc, { shouldValidate: true });
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
            <div className="grid gap-3 sm:grid-cols-2">
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
                    <Select value={field.value} onValueChange={field.onChange}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {LANGUAGES.map(([code, name]) => (
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
                <Input placeholder="C major" {...form.register("key_scale")} />
              </div>
              <div className="space-y-1.5">
                <Label>Maatsoort</Label>
                <Input placeholder="4/4" {...form.register("time_signature")} />
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
                        {QUALITY_PROFILES.map(([id, label]) => (
                          <SelectItem key={id} value={id}>{label}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>
            </div>
          </FieldGroup>
          <FieldGroup title="LoRA" description="Optioneel: kies een getrainde PEFT LoRA voor deze render.">
            <LoraSelector value={values} onChange={setLoraSelection} />
          </FieldGroup>
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
                        <SelectItem value="wav32">WAV 32-bit float</SelectItem>
                        <SelectItem value="wav16">WAV 16-bit</SelectItem>
                        <SelectItem value="mp3">MP3</SelectItem>
                        <SelectItem value="flac">FLAC</SelectItem>
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>
              <div className="space-y-1.5">
                <Label>Batch size</Label>
                <Controller
                  control={form.control}
                  name="batch_size"
                  render={({ field }) => (
                    <Slider value={[field.value]} min={1} max={8} step={1} onValueChange={(v) => field.onChange(v[0])} />
                  )}
                />
                <p className="font-mono text-xs text-muted-foreground">{values.batch_size}× variant</p>
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
              { key: "lora_adapter_name", label: "LoRA" },
              { key: "duration", label: "Duur", format: (v) => formatDuration(Number(v) || 0) },
              { key: "bpm", label: "BPM" },
              { key: "key_scale", label: "Key" },
              { key: "vocal_language", label: "Taal" },
              { key: "quality_profile", label: "Kwaliteit" },
              { key: "tags", label: "Tags" },
              { key: "inference_steps", label: "Steps" },
              { key: "batch_size", label: "Batch" },
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
        return (
          <div className="space-y-4">
            {audioUrl && (
              <WaveformPlayer
                src={audioUrl}
                title={title}
                artist={artist}
                metadata={{
                  model: lastResult.song_model ?? values.song_model,
                  quality: values.quality_profile,
                  duration: lastResult.duration ?? values.duration,
                  bpm: lastResult.bpm ?? values.bpm,
                  key: lastResult.key_scale ?? values.key_scale,
                  seed: lastResult.seed ?? values.seed,
                  resultId,
                }}
              />
            )}
            <ArtGenerator
              scope="single"
              attachToResultId={resultId}
              title={title}
              caption={(lastResult.caption as string | undefined) ?? values.caption}
              defaultPrompt={defaultTrackArtPrompt({
                title,
                artist_name: artist,
                caption: (lastResult.caption as string | undefined) ?? values.caption,
                tags: (lastResult.tags as string | undefined) ?? values.tags,
              })}
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
              <Button variant="ghost" onClick={() => setStep(0)}>Nieuwe track</Button>
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
